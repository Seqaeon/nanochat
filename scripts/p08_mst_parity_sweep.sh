#!/usr/bin/env bash
# ============================================================================
# P08: dense-parity fixes for MST (Stage 12, G1/G2/G3).
#
# WHY THIS SWEEP EXISTS
#   The paper's Pareto claim was measured against a dense arm taken from the
#   nanochat leaderboard rather than from our own runs. Our dense is uniformly
#   0.0376 bpb better, and refitting flips the result: dense wins on FLOPs/token
#   (0.86x) and training FLOPs (0.80x). See LEARNINGS.md, 2026-08-08.
#
#   At matched FLOPs the two arms also carry matched matrix parameters
#   (MST L=32: 511.8M @ 4.008e9; dense L=22: 523.4M @ 3.694e9), which is forced,
#   since FLOPs/token ~ 2 x matrix params either way. So the only question left is
#   whether block-diagonal-plus-coupling beats unstructured dense at equal params
#   and equal FLOPs. Today it loses by 0.0096 bpb. We need ~0.03 bpb back.
#
#   These three fixes are each FLOP-neutral, verified on meta device
#   (tests/test_mst_parity_fixes.py):
#     G1  --mst-sub-head-dim   per-stream head_dim, heads derived so qkv_dim == d.
#                              MST's implicit head_dim is 32; dense uses 128.
#                              Table 7's own control prices d_h 32 vs 128 at
#                              +0.0168 bpb, and d_h=32 is also the 2.13x kernel
#                              penalty in Section 7.
#     G2  --mst-final-norm     RMSNorm before lm_head, which dense does and MST
#                              never did: lm_head was fed a raw linear output.
#     G3  --mst-per-stream-ve  each stream reads its own slice of an (N*d)-wide
#                              value-embedding table, instead of all N streams
#                              receiving the same d-wide vector.
#
# ORDERING IS DELIBERATE, as in p32. The combined arm runs FIRST. If every fix
# together does not beat the baseline, the decomposition is not worth 3 GPU-hours
# and you want that answer in ~15 minutes. The singles only earn their cost once
# the combination shows signal.
#
# COST NOTE ON G3. It widens the VE tables 4x: at L=8 that is 16.8M -> 67M of
# lookup, at L=32 it is 268M -> 1.07B. FLOPs do not move, but total parameters
# and optimizer memory do. Since MST's remaining total-params "win" was only ever
# the VE table size, that axis is being traded away knowingly.
#
#   bash scripts/p08_mst_parity_sweep.sh                 # default depth 8, all groups
#   bash scripts/p08_mst_parity_sweep.sh 8               # depth positionally, as in p07
#   bash scripts/p08_mst_parity_sweep.sh 8 16            # several depths, in order
#   bash scripts/p08_mst_parity_sweep.sh --seeds 1 8     # fast first pass
#   bash scripts/p08_mst_parity_sweep.sh --group combo 8 # one group
#   bash scripts/p08_mst_parity_sweep.sh --group d16     # confirm at L=16
#   bash scripts/p08_mst_parity_sweep.sh --force 8       # ignore completion state
#
# DEPTH=8 as an environment variable still works, for consistency with p32.
#
# Reference points to beat (single seed, current ladder):
#   MST full recipe   L=8  1.0510    L=16  0.8810
#   dense d_h=128     L=8  0.9592 +- 0.0002
#   dense d_h=32      L=8  0.9760 +- 0.0003
# ============================================================================
set -o pipefail

FORCE=0
RUN_GROUPS="control combo g1 g2 g3 best overhead mix"
SEEDS=1
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)  FORCE=1; shift ;;
        --group)  RUN_GROUPS="$2"; shift 2 ;;
        --seeds)  SEEDS="$2"; shift 2 ;;
        # Bare integers are depths, as in p07. Several may be given.
        [0-9]*)   DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"; echo "usage: $0 [--force] [--group G] [--seeds N] [DEPTH ...]"; exit 1 ;;
    esac
done
# Fall back to the DEPTH env var, then 8, so p32-style invocation still works.
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=("${DEPTH:-8}")

N_SUBS="${N_SUBS:-4}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"

# G1 needs sub_dim divisible by the target head_dim. At the depths this sweep
# uses that is automatic (L=8 -> d=128, L=16 -> d=256), but most of the ladder is
# not: with D rounded to 128, only L in {8,16,20,24,32} admit head_dim 64 and
# {8,16,24,32} admit 128. Rounding D to 256 instead makes 64 legal at every
# depth, at the cost of moving D at L=9/18/22/26. Decide that before the ladder.
check_divisible() {                       # check_divisible <sub_dim> <head_dim>
    if (( $1 % $2 != 0 )); then
        echo "⚠  skipping head_dim=$2: sub_dim $1 is not divisible by it"
        return 1
    fi
    return 0
}

OUT_BASE="${OUT_BASE:-out/p08_mst_parity}"
mkdir -p "$OUT_BASE"

# The helpers below read $STATE, $COMMON, $LOGFILE and $SEEDS at call time, so
# they are defined once and pick up whatever the current depth iteration set.
done_already() {
    [ "$FORCE" -eq 1 ] && return 1
    python3 -c "
import json,sys
sys.exit(0 if '$1' in json.load(open('$STATE')).get('completed',{}) else 1)" 2>/dev/null
}
mark_done() {
    python3 -c "
import json,datetime
s=json.load(open('$STATE'))
s.setdefault('completed',{})['$1']=datetime.datetime.now().isoformat()
json.dump(s,open('$STATE','w'),indent=2)"
}

# Identical to p32's COMMON so the numbers are comparable to Table 7 directly.
COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-tokens -1 --target-active-params 0 \
  --save-every 200 --eval-every -1"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

# mst_config() emits the partition flags for a given depth's dimensions.
mst_config() {                            # mst_config <sub_dim> <n_subs>
    echo "--use-mst 1 --models base --mst-n-subs $2 --mst-sub-dim $1 \
      --mst-head-dim 0 --mst-input-mode learned_proj \
      --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
      --mst-transition-mode aggregate_distribute \
      --mst-final-mode concat_proj --mst-final-topk 0 \
      --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
      --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
      --mst-transition-width-mult $2.0 --mst-sub-lr-scale 2.0 \
      --mst-multi-scale-windows 1"
}

# Unlike p32, seeds are passed explicitly so the arms are reproducible. Note that
# --seed controls weight init only, not dataloader order, so the spread across
# seeds here is init variance (p32 measured that at sigma <= 0.0003 bpb at L=8),
# not full run-to-run variance.
run() {                                   # run <tag> <depth> <flags...>
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "⏭  $t"; continue; fi
        echo ""
        echo "━━━ $t  (depth $depth) ━━━"
        # Scoped by depth, not just by tag: tags repeat across depths, and the
        # d16 group runs at 16 from inside a depth-8 sweep. This also lands
        # base_train's mst_results.csv in a per-depth directory.
        local dir="${OUT_BASE}/d${depth}/${t}"
        [ "$FORCE" -eq 1 ] && rm -rf "$dir"
        if bash scripts/research_sweep.sh $COMMON --out-dir "$dir" --seed "$s" \
               "$@" "$depth" 2>&1 | tee -a "$LOGFILE"; then
            mark_done "$t"; echo "✅  $t"
        else
            echo "❌  $t failed; will retry on the next invocation"
        fi
    done
}

has() { echo " $RUN_GROUPS " | grep -q " $1 "; }

# ════ per-depth body ════════════════════════════════════════════════════════
for DEPTH in "${DEPTHS[@]}"; do

MODEL_DIM=$(( ((DEPTH * ASPECT_RATIO + 127) / 128) * 128 ))
SUB_DIM=$(( MODEL_DIM / N_SUBS ))
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/p08_d${DEPTH}.log}"
STATE="${OUT_BASE}/p08_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# The paper's proposed model at this depth: the arm every delta is measured from.
MST_FULL="$(mst_config "$SUB_DIM" "$N_SUBS")"

echo "════════════════════════════════════════════════════════════"
echo "  P08 MST dense-parity fixes (G1/G2/G3)"
echo "  depth ${DEPTH}   D=${MODEL_DIM}   N=${N_SUBS}   d=${SUB_DIM}"
echo "  seeds ${SEEDS}   groups: ${RUN_GROUPS}"
echo "  out ${OUT_BASE}"
echo "════════════════════════════════════════════════════════════"

# ---------------------------------------------------------------- control
# The anchor. Without a same-sweep baseline the deltas are not readable, because
# the reference 1.0510 is a single seed from a different sweep.
# The two dense controls duplicate p32's; skip this group if you already have them.
#if has control; then
#    echo ""; echo "### CONTROL: anchors for every delta below"
#    run CTRL_mst_full   "$DEPTH" $MST_FULL
#    run CTRL_dense_hd128 "$DEPTH" --models base
#    run CTRL_dense_hd32  "$DEPTH" --models base --head-dim 32
#fi
#
# ---------------------------------------------------------------- combo
# Runs before the singles on purpose: this is the go/no-go. G1 is carried at both
# 64 and 128 because at L=8 (d=128) head_dim=128 leaves exactly one head per
# stream, so the "wider heads" and "enough heads" effects point opposite ways and
# a single choice would confound them.
#if has combo; then
#    echo ""; echo "### COMBO: all three fixes (the go/no-go)"
#    if check_divisible "$SUB_DIM" 64; then
#        run COMBO_g1_64_g2_g3 "$DEPTH" $MST_FULL \
#            --mst-sub-head-dim 64 --mst-final-norm 1 --mst-per-stream-ve 1
#        run COMBO_g1_64_g2    "$DEPTH" $MST_FULL \
#            --mst-sub-head-dim 64 --mst-final-norm 1
#    fi
#    if check_divisible "$SUB_DIM" 128; then
#        run COMBO_g1_128_g2_g3 "$DEPTH" $MST_FULL \
#            --mst-sub-head-dim 128 --mst-final-norm 1 --mst-per-stream-ve 1
#    fi
#fi

# ---------------------------------------------------------------- g1
# Head geometry alone. Expected to be the largest of the three: Table 7's dense
# control puts d_h 32 vs 128 at 0.0168 bpb, and this is that same axis applied
# per stream at identical parameters and FLOPs.
#if has g1; then
#    echo ""; echo "### G1: per-stream head_dim (32 baseline -> 64 -> 128)"
#    for hd in 64 128; do
#        check_divisible "$SUB_DIM" "$hd" || continue
#        run "G1_hd${hd}" "$DEPTH" $MST_FULL --mst-sub-head-dim "$hd"
#    done
#fi
#

#if has g1; then
#    echo ""; echo "### G1: per-stream head_dim (32 baseline -> 64 -> 128)"
#    for hd in 128; do
#        check_divisible "$SUB_DIM" "$hd" || continue
#        run "G1_hd${hd}" "$DEPTH" $MST_FULL --mst-sub-head-dim "$hd"
#    done
#fi
#
# ---------------------------------------------------------------- g2
#if has g2; then
#    echo ""; echo "### G2: RMSNorm before lm_head"
#    run G2_final_norm "$DEPTH" $MST_FULL --mst-final-norm 1
#fi
#
# ---------------------------------------------------------------- g3
#if has g3; then
#    echo ""; echo "### G3: per-stream value embeddings"
#    run G3_per_stream_ve "$DEPTH" $MST_FULL --mst-per-stream-ve 1
#fi
#
# ---------------------------------------------------------------- best
# The FLOP-neutral arm the first pass missed. G2 measured +0.0021 at L=8 both
# alone and in combination, so it is dropped; G1 and G3 were -0.0101 and -0.0092
# and are near-additive, predicting about -0.019.
#if has best; then
#    echo ""; echo "### BEST: G1+G3, no G2"
#    if check_divisible "$SUB_DIM" 64; then
#        run BEST_g1_64_g3 "$DEPTH" $MST_FULL --mst-sub-head-dim 64 --mst-per-stream-ve 1
#    fi
#fi

# ---------------------------------------------------------------- overhead
# Stage 13. Unlike everything above, these CUT FLOPs rather than holding them
# fixed, which is the point: MST is at parity with dense per matrix parameter, so
# its entire FLOPs-axis deficit is D-proportional overhead it never had to pay.
# Measured on meta device at L=8 / L=16 / L=32: O1(D/2)+O2 is -37.9% / -24.7% /
# -12.9% FLOPs at essentially unchanged matrix parameters.
#
# BOTH REMOVE CAPACITY, so bpb is the thing being risked here and the reason
# these are measured rather than assumed. O1 factorizes the output head, which
# risks a softmax bottleneck. O2 caps the widest stream at the layer window on
# short layers, which costs long-range attention (though only to match what the
# dense baseline already does under SSSL).
if has overhead; then
    echo ""; echo "### OVERHEAD: output head (O1) and window composition (O2)"
#    run O2_compose_windows "$DEPTH" $MST_FULL --mst-compose-windows 1
#    run O1_head_half    "$DEPTH" $MST_FULL --mst-lm-head-dim $(( MODEL_DIM / 2 ))
#    run O1_head_quarter "$DEPTH" $MST_FULL --mst-lm-head-dim $(( MODEL_DIM / 4 ))
#    run O1_half_O2      "$DEPTH" $MST_FULL \
#        --mst-lm-head-dim $(( MODEL_DIM / 2 )) --mst-compose-windows 1
    # Everything that survived, together.
    if check_divisible "$SUB_DIM" 64; then
#        run STACK_all "$DEPTH" $MST_FULL \
#            --mst-sub-head-dim 64 --mst-per-stream-ve 1 \
#            --mst-lm-head-dim $(( MODEL_DIM / 2 )) --mst-compose-windows 1
        # O1 measured at +0.046 bpb for -27.7% FLOPs at L=8, which the matmul-param
        # scaling law predicts almost exactly (+0.042): the output head is capacity,
        # not overhead, so cutting it only slides along the parameter curve. O2 by
        # contrast was +0.0001 bpb for -10.2% FLOPs, i.e. genuinely free. This arm
        # keeps O2 and drops O1, and is the one expected to win.
#        run STACK_noO1 "$DEPTH" $MST_FULL \
#            --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1
    fi
fi

# ---------------------------------------------------------------- mix
# Stage 14. An MST layer is block-diagonal in the channel axis, and composing
# block-diagonal maps under a FIXED partition stays block-diagonal, so every
# cross-stream path has to squeeze through the rank-d coupling. That is why the
# twelve richer couplings all failed: they were rebuilding a D x D mixing matrix
# through a rank-d channel. Permuting the partition makes the composition itself
# mix, at zero parameters and zero FLOPs.
#
# Precedent: ShuffleNet showed grouped convs plateau without channel shuffle;
# ResNeXt keeps its 1x1 mixing convs dense; Swin alternates window offsets; Monarch
# factorizes as block-diagonal . permutation . block-diagonal.
#
# NOT the same as the existing mst_feature_cycle, which rolls by exactly d and so
# maps stream n to stream n+1 intact, changing nothing about which channels travel
# together. That negative result does not cover this.
#
# 'roll' shifts by d//2 so streams keep half their channels and specialization
# survives; 'shuffle' is maximal mixing but erases stream identity, which fights
# the multi-scale windows. Sites: between layers, between attention and FFN, or both.
if has mix; then
    echo ""; echo "### MIX: free cross-stream mixing (zero params, zero FLOPs)"
    BEST_SO_FAR="--mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1"
    if check_divisible "$SUB_DIM" 64; then
        for site in layer ffn both; do
            run "MIX_roll_${site}" "$DEPTH" $MST_FULL $BEST_SO_FAR \
                --mst-channel-mix roll --mst-channel-mix-site "$site"
        done
        # Maximal mixing, run only at the best site once 'roll' has named one.
        run MIX_shuffle_layer "$DEPTH" $MST_FULL $BEST_SO_FAR \
            --mst-channel-mix shuffle --mst-channel-mix-site layer
    fi
fi


# ---------------------------------------------------------------- d16
# Opt-in, and single seed: an L=16 grid costs roughly 40x an L=8 one. Run this
# only after L=8 names a winner, and only for that winner plus its baseline, so
# the transfer argument of Table 6 covers these fixes too.
# It always runs at L=16 regardless of the depth being swept, so skip it when the
# main groups are already at 16 rather than paying twice under a second tag.
if has d16 && [ "$DEPTH" -ne 16 ]; then
    echo ""; echo "### D16: transfer check for the L=8 winner"
    D16=16
    MD16=$(( ((D16 * ASPECT_RATIO + 127) / 128) * 128 ))
    SD16=$(( MD16 / N_SUBS ))
    MST_FULL_16="$(mst_config "$SD16" "$N_SUBS")"
    SEEDS_SAVE="$SEEDS"; SEEDS=1
    run D16_baseline "$D16" $MST_FULL_16
    # G2 dropped after the L=8 result; the full stack carries the overhead cuts too,
    # which are worth -24.7% FLOPs at this depth against -12.9% at L=32.
    run D16_g1_64_g3 "$D16" $MST_FULL_16 --mst-sub-head-dim 64 --mst-per-stream-ve 1
    # O1 dropped: it costs bpb in line with the parameter scaling law, so it is not
    # a free saving. O2 was free at L=8; whether it stays free at L=16, where long
    # context is worth more, is the main thing this arm is testing.
    run D16_stack    "$D16" $MST_FULL_16 \
        --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1
    SEEDS="$SEEDS_SAVE"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  P08 complete for depth ${DEPTH}"
echo "  results: ${OUT_BASE}/d${DEPTH}/mst_results*.csv   log: ${LOGFILE}"
echo ""
echo "  Read it as: every delta is against CTRL_mst_full in this same sweep."
echo "  The bar is ~0.03 bpb of total improvement, which is what turns the"
echo "  corrected FLOPs multiplier from 0.86x into roughly 1.15x."
echo "════════════════════════════════════════════════════════════"

done
