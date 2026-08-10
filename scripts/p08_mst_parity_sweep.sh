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
RUN_GROUPS="control combo g1 g2 g3 best overhead mix couple sparse shampoo monarch mol"
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
  --sequence-len ${SEQ_LEN:-2048} --target-param-data-ratio 10.5 \
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

# mol_config() emits the MoL baseline's flags (arXiv:2605.09516). Their notation is
# S+KofN, so mol_config 15 1 3 <d> is their headline 1+3of15: one always-active shared
# block plus top-3 of 14 routed, four active blocks per token.
mol_config() {                            # mol_config <n_blocks> <n_shared> <topk> <thin_dim>
    echo "--use-mol 1 --models base --mol-n-blocks $1 --mol-n-shared $2 \
      --mol-topk $3 --mol-thin-dim $4 --mol-head-dim 64 --mol-ffn-mult 4.0 \
      --mol-router-aux 0.05 --mol-routed-attn softmax --mol-dispatch 1"
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
#if has overhead; then
#    echo ""; echo "### OVERHEAD: output head (O1) and window composition (O2)"
#    run O2_compose_windows "$DEPTH" $MST_FULL --mst-compose-windows 1
#    run O1_head_half    "$DEPTH" $MST_FULL --mst-lm-head-dim $(( MODEL_DIM / 2 ))
#    run O1_head_quarter "$DEPTH" $MST_FULL --mst-lm-head-dim $(( MODEL_DIM / 4 ))
#    run O1_half_O2      "$DEPTH" $MST_FULL \
#        --mst-lm-head-dim $(( MODEL_DIM / 2 )) --mst-compose-windows 1
#    # Everything that survived, together.
#    if check_divisible "$SUB_DIM" 64; then
#        run STACK_all "$DEPTH" $MST_FULL \
#            --mst-sub-head-dim 64 --mst-per-stream-ve 1 \
#            --mst-lm-head-dim $(( MODEL_DIM / 2 )) --mst-compose-windows 1
#        # O1 measured at +0.046 bpb for -27.7% FLOPs at L=8, which the matmul-param
#        # scaling law predicts almost exactly (+0.042): the output head is capacity,
#        # not overhead, so cutting it only slides along the parameter curve. O2 by
#        # contrast was +0.0001 bpb for -10.2% FLOPs, i.e. genuinely free. This arm
#        # keeps O2 and drops O1, and is the one expected to win.
#        run STACK_noO1 "$DEPTH" $MST_FULL \
#            --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1
#    fi
#fi
#
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
#if has mix; then
#    echo ""; echo "### MIX: free cross-stream mixing (zero params, zero FLOPs)"
#    BEST_SO_FAR="--mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1"
#    if check_divisible "$SUB_DIM" 64; then
#        for site in layer ffn both; do
#            run "MIX_roll_${site}" "$DEPTH" $MST_FULL $BEST_SO_FAR \
#                --mst-channel-mix roll --mst-channel-mix-site "$site"
#        done
#        # Maximal mixing, run only at the best site once 'roll' has named one.
#        run MIX_shuffle_layer "$DEPTH" $MST_FULL $BEST_SO_FAR \
#            --mst-channel-mix shuffle --mst-channel-mix-site layer
#    fi
#fi

# ---------------------------------------------------------------- mixiso
# The first MIX pass was not a clean test of the mechanism, for two reasons.
#
#   1. CONFOUND. Attention windows are indexed by SLOT (window_sizes[j] against the
#      j axis of (B,T,N,d)), not by channel. Permuting the partition therefore
#      reassigns which channels get which attention scale, so MIX_roll_layer coming
#      back 5 sigma worse is as consistent with "the window specialization broke" as
#      with "mixing does not help". Preserving a channel's scale under a d/2 shift is
#      not possible while scales are per-stream: each slot then holds channels from
#      two scale groups and attention takes one window per stream. So the way to
#      separate the two effects is to remove the specialization, not to preserve it.
#
#   2. DEPTH. Shifted partitions work by composition. With a d/2 shift the reachable
#      span grows about half a block per layer, so at N=4 a channel needs roughly six
#      layers to reach all four streams. L=8 completes that once; L=32 completes it
#      four times. Testing a depth-dependent mechanism at the shallowest depth and
#      generalizing was the error.
#
# This group fixes both. Multi-scale windows OFF makes every stream share the layer
# window, so there is no specialization to scramble and the mixing arms are exactly
# FLOP-identical to their own control. The group is depth-agnostic, so the depth test
# is just: --group mixiso 16.
#
# Two seeds, because MIX_roll_ffn landed at 1.6 sigma, which is unmeasured rather
# than null. The reference arms are included and will only run their missing seed.
#if has mixiso; then
#    echo ""; echo "### MIXISO: mixing without the window confound (uniform windows)"
#    SEEDS_SAVE="$SEEDS"; SEEDS=2
#    if check_divisible "$SUB_DIM" 64; then
#        # Uniform windows: msw off (a later flag wins), so compose-windows is a no-op
#        # and every stream inherits the SSSL layer window. Costs more attention FLOPs
#        # than the multi-scale schedule, which is fine: the comparison is within this
#        # setting, and all three arms below carry identical FLOPs.
#        UNIFORM="$MST_FULL --mst-multi-scale-windows 0 \
#                 --mst-sub-head-dim 64 --mst-per-stream-ve 1"
#        run MIXISO_uni_none      "$DEPTH" $UNIFORM
#        run MIXISO_uni_roll_ffn  "$DEPTH" $UNIFORM --mst-channel-mix roll --mst-channel-mix-site ffn
#        run MIXISO_uni_roll_layer "$DEPTH" $UNIFORM --mst-channel-mix roll --mst-channel-mix-site layer
#
#        # Second seed for the two arms the first pass could not resolve, reusing the
#        # completed s1 runs. Same flags as the originals or the comparison is void.
#        BEST_SO_FAR="--mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1"
#        run STACK_noO1   "$DEPTH" $MST_FULL $BEST_SO_FAR
#        run MIX_roll_ffn "$DEPTH" $MST_FULL $BEST_SO_FAR \
#            --mst-channel-mix roll --mst-channel-mix-site ffn
#    fi
#    SEEDS="$SEEDS_SAVE"
#fi


# ---------------------------------------------------------------- couple
# Stage 15. Five mechanisms, all stacked on STACK_noO1 (which already has 2 seeds and
# serves as the control at no cost).
#
#   F1 --mst-distribute-block-muon  distribute_w is (N*d, d), structurally identical to
#      c_proj_w, but was omitted from setup_optimizer's stacked_names. Muon therefore
#      orthogonalizes across all N coupling blocks jointly -- the exact cross-contamination
#      that 1B exists to remove -- and it never gets the sub-LR correction. The coupling
#      was the only part of the model still optimized as if it were one dense matrix, which
#      is a candidate explanation for why all twelve coupling enrichments hit equifinality.
#   F2 --mst-trans-spectral-lr      agg_up (D,d) and agg_down (d,D) are the only weights
#      whose fan_out/fan_in is not 1. Under Muon's spectral normalization they want LRs
#      differing by N; today they share one matrix_lr.
#   F3 --mst-transition-every       couple every k-th layer (last always couples). The
#      coupling is ~20% of a layer and our own data says it saturates.
#   F4 --mst-talking-heads          learned (N*n_head)^2 mixing along the head axis before
#      c_proj. Dense MHA is ALSO block-diagonal per head and works because W_O mixes them;
#      MST blocks W_O too. Measured +0.006% params, +0.005% FLOPs: effectively free.
#   F5 --mst-wo-mode dense          the full version of F4's mechanism. Measured +19.7%
#      params / +16.3% FLOPs at L=32, +5.7% FLOPs at L=8.
#
# SCORING. Use sigma = 0.00184 (measured over five 2-seed arms), NOT the 0.0003 in the
# paper, which measures kernel nondeterminism because p32 never passed --seed. Resolution
# floor at 2 seeds is 0.0037 bpb. F5 must buy >0.0058 bpb at L=8 to pay for its FLOPs.
# F3's saving is depth-dependent (-2.1% FLOPs at L=8 but -7.6% at L=32), so judge it at L=8
# on whether it COSTS bpb, not on the L=8 FLOP payoff.
#if has couple; then
#    echo ""; echo "### COUPLE: Stage 15 coupling optimization + attention mixing"
#    SEEDS_SAVE="$SEEDS"; SEEDS=2
#    if check_divisible "$SUB_DIM" 64; then
#        BEST_SO_FAR="--mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1"
#        # Free and possibly explanatory, so these run first.
#        run CPL_distmuon           "$DEPTH" $MST_FULL $BEST_SO_FAR --mst-distribute-block-muon 1
#        run CPL_talking            "$DEPTH" $MST_FULL $BEST_SO_FAR --mst-talking-heads 1
#        run CPL_spectral           "$DEPTH" $MST_FULL $BEST_SO_FAR --mst-trans-spectral-lr 1
#        run CPL_distmuon_spectral  "$DEPTH" $MST_FULL $BEST_SO_FAR \
#            --mst-distribute-block-muon 1 --mst-trans-spectral-lr 1
#        run CPL_talking_distmuon   "$DEPTH" $MST_FULL $BEST_SO_FAR \
#            --mst-talking-heads 1 --mst-distribute-block-muon 1
#        # Cost cuts.
#        run CPL_every2             "$DEPTH" $MST_FULL $BEST_SO_FAR --mst-transition-every 2
#        run CPL_every4             "$DEPTH" $MST_FULL $BEST_SO_FAR --mst-transition-every 4
#        # If F1 fixes how the coupling is trained, coupling less often may cost less.
#        run CPL_every2_distmuon    "$DEPTH" $MST_FULL $BEST_SO_FAR \
#            --mst-transition-every 2 --mst-distribute-block-muon 1
#        # The expensive arm.
#        run CPL_dense_wo           "$DEPTH" $MST_FULL $BEST_SO_FAR --mst-wo-mode dense
#        # Does the free stuff stack onto the winner? F5 was -0.0107 at 4.5 sigma; F4 and
#        # F1 are each under 1 sigma alone but point the same way and add up (-0.0025
#        # together), and both are exactly FLOP-free, so there is no downside to carrying
#        # them if they hold.
#        run CPL_dense_wo_stack     "$DEPTH" $MST_FULL $BEST_SO_FAR \
#            --mst-wo-mode dense --mst-talking-heads 1 --mst-distribute-block-muon 1
#        # Control, reusing its completed seeds. Flags must stay byte-identical.
#        run STACK_noO1             "$DEPTH" $MST_FULL $BEST_SO_FAR
#    fi
#    SEEDS="$SEEDS_SAVE"
#fi

# ---------------------------------------------------------------- sparse
# Stage 16: conditional stream execution. Everything up to here bought quality by adding
# parameters, which is capped by the parameter scaling law -- the same law dense obeys, so
# it cannot beat dense. (Dense W_O beat that law by only +0.0030 bpb.) Sparsity is a
# different axis: k of N streams per token, so active FLOPs fall below total.
#
# Phase A is compute-then-mask. It does not make the model faster; it prices what k-of-N
# sparsity COSTS in bpb, which is the whole research question. The gather/scatter dispatch
# that realises the saving is only worth building if this comes back cheap.
#
# Measured active-FLOP savings on the current best arm (FFN gating):
#            L=16      L=32
#   k=3     -6.9%     -9.3%
#   k=2    -13.8%    -18.6%
#   k=1    -20.8%    -28.0%
# Multipliers at L=16 IF bpb held: k=3 1.10x, k=2 1.19x, k=1 1.30x, against 1.028x today.
# Quality budget to break even at L=16: 0.013 bpb at k=2, 0.020 at k=1. The 2-sigma floor
# is 0.0048, so all of these are comfortably measurable.
#
# SP_k2_noaux exists because the prior routing attempt (free_for_all + topk1) collapsed to
# uniform on replicate seeds (router_entropy = log(8), load_balance = 1.0). If the aux loss
# is what prevents that, this arm should be visibly worse or unstable across its two seeds.
if has sparse; then
    echo ""; echo "### SPARSE: conditional stream execution (Stage 16, Phase A)"
    SEEDS_SAVE="$SEEDS"; SEEDS=1
    if check_divisible "$SUB_DIM" 64; then
        BEST="--mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 --mst-wo-mode dense"
        # ALL the SP_* results are invalid and these are retagged v2. The router they
        # measured had three defects, found by wiring stream_load into compute_diagnostics
        # (route_entropy_* is the TRANSITION router and says nothing about the gate):
        #
        #   1. Zero-init WAS the collapsed state. Every logit at exactly 0, topk breaking
        #      ties by index, so the same k streams won for every token before training,
        #      and the losers' FFNs got exactly zero gradient. Now small random init.
        #   2. The aux loss was not well posed. Switch's N*sum(f_i*P_i) needs both factors
        #      on the simplex; independent sigmoids have no such constraint, so it was
        #      minimized by pushing every gate to zero. It balanced nothing and cost
        #      +0.0105 bpb. The gate is a softmax now and the term is correct.
        #   3. No exploration. An unselected stream gets no FFN gradient, stays at init,
        #      stays unselected. Noisy top-k breaks that spiral.
        #
        # Measured over 300 steps at depth 4 (worst-layer min load / ideal, higher better):
        #   correct aux, no noise   0.273     aux + noise 0.3   0.484
        #   aux + noise 1.0         0.586     NO AUX + noise    0.000, 5/16 streams dead
        #
        # So the aux term is necessary after all; it was the formulation that was wrong.
        # Note the no-aux arm had the LOWEST training loss while being the most collapsed,
        # which is exactly why the load diagnostic has to be read alongside bpb.
#        run SP2_k2         "$DEPTH" $MST_FULL $BEST --mst-stream-topk 2 --mst-stream-router-noise 1.0
        run SP2_k1         "$DEPTH" $MST_FULL $BEST --mst-stream-topk 1 --mst-stream-router-noise 1.0
#        run SP2_k3         "$DEPTH" $MST_FULL $BEST --mst-stream-topk 3 --mst-stream-router-noise 1.0
#        # Isolate the two mechanisms.
#        run SP2_k2_nonoise "$DEPTH" $MST_FULL $BEST --mst-stream-topk 2 --mst-stream-router-noise 0.0
#        run SP2_k2_noaux   "$DEPTH" $MST_FULL $BEST --mst-stream-topk 2 --mst-stream-router-noise 1.0 \
#            --mst-stream-router-aux 0
#        # Control, reusing its completed seeds. Flags must stay byte-identical.
#        run CPL_dense_wo "$DEPTH" $MST_FULL $BEST
    fi
    SEEDS="$SEEDS_SAVE"
fi
#
# ---------------------------------------------------------------- shampoo
# Stage 17: block-diagonal Shampoo. This is the architecture claim expressed as an
# optimizer. Preconditioning a dense D x D weight costs O(D^3); MST's stacked per-stream
# weights are N blocks of d x d, so exact block preconditioning is D^3/N^2 -- 16x cheaper
# at N=4, verified in tests/test_shampoo.py. K-FAC and Shampoo both *approximate*
# block-diagonality; MST makes it exact by construction, so at equal optimizer cost MST
# affords a preconditioner dense cannot.
#
# HOW TO SCORE THIS HONESTLY. Shampoo does not change the model, so it is free in
# estimate_flops and any bpb gain reads as pure Pareto movement. A reviewer will object,
# correctly. Record the step wall-clock (the tok/sec column in the log) alongside bpb: the
# claim is "MST can afford this preconditioner and dense cannot", not "this is free".
#if has shampoo; then
#    echo ""; echo "### SHAMPOO: block-diagonal preconditioning (Stage 17)"
#    SEEDS_SAVE="$SEEDS"; SEEDS=1
#    if check_divisible "$SUB_DIM" 64; then
#        BEST="--mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 --mst-wo-mode dense"
#        # The first shampoo pass was invalid: the update was normalized to ||g||_F rather
#        # than to Muon's semi-orthogonal ~sqrt(min(m,n)), so the effective LR was ~3e4 too
#        # small and all three cadences landed within 0.004 bpb of each other at +0.07 vs
#        # control. Fixed in optim.py (shampoo_step) and pinned by
#        # tests/test_shampoo.py::test_update_norm_matches_muons_convention_at_any_gradient_scale.
#        # These tags are v2 so the invalid results are not silently reused.
#        run SH2_every10 "$DEPTH" $MST_FULL $BEST --mst-shampoo 1 --mst-precond-every 10
#        run SH2_every1  "$DEPTH" $MST_FULL $BEST --mst-shampoo 1 --mst-precond-every 1
#        run SH2_every50 "$DEPTH" $MST_FULL $BEST --mst-shampoo 1 --mst-precond-every 50
#        # Control, reusing its completed seeds. Flags must stay byte-identical.
#        run CPL_dense_wo "$DEPTH" $MST_FULL $BEST
#    fi
#    SEEDS="$SEEDS_SAVE"
#fi
#
# ---------------------------------------------------------------- monarch
# Stage 18. The FFN is already two thirds of a Monarch factorization: fc_w (per-stream
# d->4d) then fc_proj_w (per-stream 4d->d) is block-diagonal . permutation . block-diagonal
# with the permutation set to IDENTITY. Permuting the N*4d hidden axis between them makes
# it a real Monarch matrix, so each stream's down-projection reads hidden units produced by
# other streams' up-projections. Zero parameters, zero FLOPs, so any gain is pure Pareto.
#
# This is a DIFFERENT placement from the Stage 14 null result, which permuted the stream
# axis BETWEEN layers as a change of basis and undid it. This permutes the hidden axis
# INSIDE one FFN and never inverts, which is what makes the FFN itself Monarch rather than
# a composition of two independently block-diagonal maps.
#
# 'shuffle' is the true transpose (every stream draws 4d/N units from every other);
# 'roll' only trades with one neighbour. Measured structurally in the tests.
#
# CALIBRATION. The equal-depth gap to dense at L=16 is 0.0714 bpb, which is 6.6x the total
# of every architectural win so far -- this will not close it. Realistic prize is
# 0.005-0.015. The one measured structured-matrix result in this repo is negative
# (Kronecker 18C: +0.12 loss, "too constrained to represent dense W").
#
# MON_shuffle_k1 IS NOT DEPLOYABLE. Monarch needs every stream's up-projection to exist,
# and conditional stream execution's whole saving is not computing them, so the two are
# alternatives rather than a stack: --mst-stream-dispatch is asserted incompatible, and
# masked sparsity only halves the saving. That arm exists solely to check whether the two
# mechanisms fight, since routing pays when streams are differentiated and mixing makes
# them interchangeable. Score the winner against the k=1 sparse arm (1.194x at L=16),
# not just against the control.
#if has monarch; then
#    echo ""; echo "### MONARCH: hidden-axis permutation inside the FFN (Stage 18)"
#    SEEDS_SAVE="$SEEDS"; SEEDS=2
#    if check_divisible "$SUB_DIM" 64; then
#        BEST="--mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 --mst-wo-mode dense"
#        run MON_shuffle    "$DEPTH" $MST_FULL $BEST --mst-ffn-monarch shuffle
#        run MON_roll       "$DEPTH" $MST_FULL $BEST --mst-ffn-monarch roll
#        run MON_shuffle_k1 "$DEPTH" $MST_FULL $BEST --mst-ffn-monarch shuffle \
#            --mst-stream-topk 1 --mst-stream-router-noise 1.0
#        # Control, reusing its completed seeds. Flags must stay byte-identical.
#        run CPL_dense_wo "$DEPTH" $MST_FULL $BEST
#    fi
#    SEEDS="$SEEDS_SAVE"
#fi

# ---------------------------------------------------------------- d16
# Opt-in, and single seed: an L=16 grid costs roughly 40x an L=8 one. Run this
# only after L=8 names a winner, and only for that winner plus its baseline, so
# the transfer argument of Table 6 covers these fixes too.
# Its arms are pinned to L=16 regardless of the depth being swept, so `--group d16`
# does the same thing from a depth-8 or a depth-16 invocation. It is opt-in (not in the
# default RUN_GROUPS), so it never runs by accident.
#
# There used to be a `[ "$DEPTH" -ne 16 ]` guard here to avoid training the same config
# twice when the main groups were also at 16 (D16_stack is the same config as
# STACK_noO1). That silently turned `--group d16 16` into a no-op that still printed a
# success banner. Duplication is now the caller's problem and is visible in the arm
# list; a flag that quietly does nothing is not.
if has d16; then
    echo ""; echo "### D16: transfer check for the L=8 winner"
    D16=16
    MD16=$(( ((D16 * ASPECT_RATIO + 127) / 128) * 128 ))
    SD16=$(( MD16 / N_SUBS ))
    MST_FULL_16="$(mst_config "$SD16" "$N_SUBS")"
    SEEDS_SAVE="$SEEDS"; SEEDS=1
#    run D16_baseline "$D16" $MST_FULL_16
    # G2 dropped after the L=8 result; the full stack carries the overhead cuts too,
    # which are worth -24.7% FLOPs at this depth against -12.9% at L=32.
#    run D16_g1_64_g3 "$D16" $MST_FULL_16 --mst-sub-head-dim 64 --mst-per-stream-ve 1
    # O1 dropped: it costs bpb in line with the parameter scaling law, so it is not
    # a free saving. O2 was free at L=8; whether it stays free at L=16, where long
    # context is worth more, is the main thing this arm is testing.
#    run D16_stack    "$D16" $MST_FULL_16 \
#        --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1
    # THE decisive arm. F5 bought 0.0107 bpb at L=8, but its cost grows with depth and
    # is charged twice on the training-FLOPs axis (more FLOPs per token, and a larger
    # token budget because tokens = 10.5 x scaling params). It must buy >0.0097 bpb here
    # and >0.0113 at L=32. Compare against D16_stack, which differs only by --mst-wo-mode.
    #   full transfer of the L=8 gain -> 1.0008x        (a win)
    #   the 0.26 transfer G1+G3 showed -> 0.9141x       (worse than doing nothing)
    # One single-seed run separates those, which is why it is worth its cost.
#    run D16_dense_wo "$D16" $MST_FULL_16 \
#        --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 \
#        --mst-wo-mode dense

    # ── Is G3 still worth its parameters at L=16? ──
    # G3 (--mst-per-stream-ve) widens the value-embedding table to N*d: +201M params at
    # L=16 and +805M at L=32, for a gain measured at -0.0092 bpb at L=8 that decayed
    # 0.26x by L=16, i.e. to ~-0.0024, under the 0.0048 two-sigma floor. It is a lookup,
    # so it costs zero FLOPs, which is why it survived this long: on the FLOPs-vs-bpb
    # axis it is free quality. On the total-params axis it is 44% of the model at L=32,
    # and it threw away MST's genuine advantage there (plain VE sits at d = D/4 where
    # dense's sits at D, so MST's is 4x cheaper than dense's until G3 undoes that).
    #
    # ══ MEASURED ON TOP OF SP2_k1, NOT D16_dense_wo. ══
    # Not merely for consistency with the headline config. G3 gives each stream its own
    # value-embedding vector, which is a STREAM-DIFFERENTIATION signal, and conditional
    # execution is precisely the mechanism that pays for differentiation: the Monarch
    # result showed routing and cross-stream mixing actively trade off (mixing helped
    # -0.00085 alone but cost +0.00082 on top of sparsity). So G3 may well be worth MORE
    # under k=1 routing than without it, and the -0.0092 -> -0.0024 decay that motivates
    # dropping it was measured on the NON-sparse config. Ablating it against a config we
    # no longer ship would answer a question we are not asking.
    # Note --mst-stream-gate-attn is off, so attention (where VE is injected) runs for
    # every stream even at k=1; sparsity gates the FFN. The two are not trivially
    # independent, which is the whole reason to measure rather than assume.
    #
    # Four arms, differing only in VE treatment, all carrying --mst-stream-topk 1:
    #   D16_sp2_k1        the control, byte-identical flags to the SP2_k1 arm
    #   D16_k1_ve_plain   drop G3.            -201M params,  +0.000% FLOPs
    #   D16_k1_ve_map     full d x d map.     -199M params,  +1.735% FLOPs
    #   D16_k1_ve_map_r32 identity + rank-32. -201M params,  +0.434% FLOPs
    #
    # The control is re-run in-sweep rather than compared against the existing d16
    # SP2_k1 number, because reading deltas against a differently-sourced baseline is the
    # exact error that invalidated this project's original Pareto claim (LEARNINGS.md,
    # 2026-08-08). If you are confident the existing SP2_k1 d16 run used identical flags,
    # comment it out and save one L=16 run.
    #
    # The map keeps ONE d-wide table and gives each stream its own learned view of it,
    # which is strictly less expressive than G3 (all N vectors are linear images of one
    # shared vector rather than N independent lookups). Whether that is enough is the
    # question. Both map forms are identity at init, so they start exactly at
    # D16_k1_ve_plain and can only differ by what they learn.
    #
    # HOW TO READ IT. A FLOPs increase has to pay for itself: with the dense exponent
    # -0.1004, +1.735% needs ~0.0015 bpb and +0.434% needs ~0.0004 bpb just to break
    # even on the Pareto curve. So rank-32 is the arm most likely to be a real win, and
    # the full map is close to needing all of G3's benefit back to justify itself.
    # If D16_k1_ve_plain is within 0.0048 of D16_sp2_k1, G3 is noise even under routing
    # and the simplest answer is to drop it everywhere and bank the 201M.
    K1="--mst-stream-topk 1 --mst-stream-router-noise 1.0"
#    run D16_sp2_k1        "$D16" $MST_FULL_16 \
#        --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 \
#        --mst-wo-mode dense $K1
    run D16_k1_ve_plain   "$D16" $MST_FULL_16 \
        --mst-sub-head-dim 64 --mst-compose-windows 1 --mst-wo-mode dense $K1
    run D16_k1_ve_map     "$D16" $MST_FULL_16 \
        --mst-sub-head-dim 64 --mst-compose-windows 1 --mst-wo-mode dense $K1 \
        --mst-ve-map 1
    run D16_k1_ve_map_r32 "$D16" $MST_FULL_16 \
        --mst-sub-head-dim 64 --mst-compose-windows 1 --mst-wo-mode dense $K1 \
        --mst-ve-map 1 --mst-ve-map-rank 32
    SEEDS="$SEEDS_SAVE"
fi

# ---------------------------------------------------------------- d32
# THE paper-deciding run. The iso-quality multiplier for the k=1 sparse arm is
# 0.804x at L=8 and 1.194x at L=16: it crosses 1.0 somewhere between them and grows
# 1.49x across one doubling. Whether that is a trend or a two-point coincidence is the
# single highest-information thing left to measure, because the whole claim is that
# MST's advantage GROWS with scale rather than being a small-model artifact. Two points
# cannot distinguish a trend from a line through noise; three can.
#
# Deliberately NOT passing --mst-stream-dispatch, so the curve stays apples-to-apples
# with d8 and d16 (both ran the masked Phase A path, dispatch defaults to 0). This costs
# nothing scientifically: estimate_flops() already discounts active_flops for stream
# sparsity on the masked path, so the bpb-vs-FLOPs point is the real one. Dispatch turns
# that FLOPs saving into wall-clock and is an engineering question, not part of the claim.
#
# Opt-in and single seed, like d16. An L=32 run is roughly 40x an L=16 one, so this is
# two arms and no exploration: the winner and the control it has to beat.
if has d32; then
    echo ""; echo "### D32: does the Pareto multiplier keep growing with scale?"
    D32=32
    MD32=$(( ((D32 * ASPECT_RATIO + 127) / 128) * 128 ))
    SD32=$(( MD32 / N_SUBS ))
    MST_FULL_32="$(mst_config "$SD32" "$N_SUBS")"
    SEEDS_SAVE="$SEEDS"; SEEDS=1
    if check_divisible "$SD32" 64; then
        BEST32="--mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 --mst-wo-mode dense"
        # Control: the best non-sparse config, so the sparse delta is isolated at this depth.
#        run D32_dense_wo "$D32" $MST_FULL_32 $BEST32
        # The arm the paper rests on.
        run D32_sp2_k1   "$D32" $MST_FULL_32 $BEST32 \
            --mst-stream-topk 1 --mst-stream-router-noise 1.0
    fi
    SEEDS="$SEEDS_SAVE"
fi

# ---------------------------------------------------------------- mol
# MoL's topology at the depth being swept, which is where it is actually affordable.
# d_thin is set to SUB_DIM = MODEL_DIM/4, so MoL's own design rule from their Appendix I
# (K_active x d_expert ~ d_model) is satisfied automatically at every depth, and the
# active width matches MST's N=4 streams exactly.
#
# ══ THE TOKEN BUDGET IS DELIBERATELY NOT ADJUSTED FOR MoL. ══
# Every arm here gets 10.5 x (transformer_matrices + lm_head), the same rule dense and
# MST get, with --target-active-params 0. MoL draws a bigger budget than MST only
# because it genuinely has ~3.7x more transformer matrices. base_train.py:1509 offers
# --target-active-params 1, which would discount its 14 inactive blocks and cut MoL's
# budget ~3.2x, and we do NOT use it, for three reasons:
#   1. Their Appendix J pre-registers the objection: at 20B tokens MoL saw "only 9.6
#      tokens/param, well below Chinchilla-optimal (~20x)", and names undertraining as a
#      candidate cause of their own iso-total gap. Starving them further walks into it.
#   2. It is not symmetric in effect: MoL loses 3.2x, MST_sp2_k1 only 1.3x, dense none.
#   3. The Pareto axis already prices it. A larger token budget IS larger training FLOPs,
#      and the claim is bpb-vs-FLOPs. Cutting tokens as well as charging the FLOPs would
#      count the same penalty twice.
# MoL is free to spend more compute; it just has to buy proportionate bpb.
#
# MoL's compute disadvantage COMPOUNDS with scale (training FLOPs vs MST_sp2_k1 at the
# same geometry): 1.89x at L=8, 3.44x at L=16, 4.95x at L=24. So L=8 is the conservative
# place to test this, not a concession: if MST wins here it wins by more at 1.3B.
if has mol; then
    echo ""; echo "### MOL: their topology at this depth, against MST's best"
    BEST_K1_BASE="--mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 --mst-wo-mode dense"
    if check_divisible "$SUB_DIM" 64; then
        # Their headline topology: 1 shared softmax + top-3 of 14 routed = 4 active.
        run MOL_1plus3of15 "$DEPTH" $(mol_config 15 1 3 "$SUB_DIM")
#        # Same, with the G3 EQUIVALENT: each thin block reads its own value-embedding
#        # slice instead of all 15 sharing one vector. Without this MoL is handicapped by
#        # exactly the component we measured to be worth 0.0059 bpb to MST at L=16, so
#        # both arms are run and MoL's better one is what gets reported.
#        # The cost is not symmetric and that is a real architectural consequence, not
#        # unfairness: MST's per-stream table is N*d = D wide, MoL's per-block table is
#        # n_blocks * d_thin = 3.75x D, and it takes MoL from 89.7M to 324.6M at L=8.
#        run MOL_1plus3of15_ve "$DEPTH" $(mol_config 15 1 3 "$SUB_DIM") --mol-per-block-ve 1

        # ── MST at MoL's topology ──
        # MST's streams PARTITION the residual, so N*d = D and N is bounded by D. That
        # bound is loose: what is actually constrained is head_dim, because G1's
        # FLOP-neutral form needs head_dim to DIVIDE d (qkv_dim == d). So finer N is
        # always reachable, at the price of narrower heads:
        #     L=8, D=512:  N=4 -> d=128 hd=64 | N=8 -> d=64 hd=64 | N=16 -> d=32 hd=32
        # Getting head_dim=64 at d=32 would mean expanding qkv beyond d, which is
        # exactly what G1 exists to avoid. mst_head_dim can do it and it is not
        # FLOP-neutral, so it is not used here.
        #
        # MoL has no analogous constraint: each block owns a projection pair, so
        # n_blocks and d_thin are independent. That freedom is what its 40-73% of
        # wrapper parameters buys.
        #
        # THE CONFOUND, and it is the reason both N=8 and N=16 are run. Going finer at
        # fixed k/N is FLOPs-cheaper, because MST's coupling is (N+8)/(13N+8) of a layer
        # and falls toward 1/13. But past N=8 at L=8 it also forces head_dim from 64 down
        # to 32, giving back G1, which was worth -0.0101 bpb. Rough arithmetic at the
        # dense exponent -0.1004: N=16 saves 12.9% of FLOPs (worth ~1.15x) and costs
        # ~0.0101 bpb (worth ~0.90x), so it is nearly a wash and has to be measured.
        # N=8 has no such confound: it keeps head_dim=64 and still saves 8.7%.
        #
        # --mst-stream-shared is the "1+" of S+KofN. MST never gates attention, so it has
        # no coverage problem to fix and a shared stream is pure always-on capacity.
        #
        # This is also the granularity question, on which the literature disagrees with
        # itself: DeepSeekMoE says finer is better, MoL's own Appendix I says the opposite
        # ("K=2 and K=3 tie, K=4 trails, opposite to MoE granularity findings").
#        best_for() {                      # best_for <head_dim>
#            echo "--mst-sub-head-dim $1 --mst-per-stream-ve 1 --mst-compose-windows 1 --mst-wo-mode dense"
#        }
#        pick_hd() {                       # pick_hd <sub_dim> -> largest of 64/32/16 dividing it
#            for h in 64 32 16; do
#                if [ $(( $1 % h )) -eq 0 ] && [ "$1" -ge "$h" ]; then echo "$h"; return; fi
#            done
#            echo 0
#        }
#        for MULT in 2 4; do
#            FN=$(( N_SUBS * MULT ))
#            if [ $(( MODEL_DIM % FN )) -ne 0 ]; then
#                echo "⚠  skipping N=${FN}: ${MODEL_DIM} not divisible by it"; continue
#            fi
#            FD=$(( MODEL_DIM / FN ))
#            FHD=$(pick_hd "$FD")
#            if [ "$FHD" -eq 0 ]; then
#                echo "⚠  skipping N=${FN}: sub_dim ${FD} admits no head_dim >= 16"; continue
#            fi
#            FK=$(( FN / 4 ))              # hold the active fraction k/N at 1/4
#            run "MST_n${FN}_k${FK}" "$DEPTH" $(mst_config "$FD" "$FN") $(best_for "$FHD") \
#                --mst-stream-topk "$FK" --mst-stream-router-noise 1.0
#            run "MST_n${FN}_1plus$(( FK - 1 ))of$(( FN - 1 ))" "$DEPTH" \
#                $(mst_config "$FD" "$FN") $(best_for "$FHD") \
#                --mst-stream-topk $(( FK - 1 )) --mst-stream-shared 1 \
#                --mst-stream-router-noise 1.0
#        done
        # Their Table 1 configuration: K=5 top-3, no shared block, dense FFN thin blocks.
#        run MOL_3of5       "$DEPTH" $(mol_config 5 0 3 "$SUB_DIM")
#        # All-active, to separate "narrow blocks" from "routing between them". Their
#        # Table 1 prices selective activation at 0.99 PPL over uniform composition.
#        run MOL_allactive  "$DEPTH" $(mol_config 4 4 1 "$SUB_DIM")
    fi
fi

# ---------------------------------------------------------------- m13
# MoL at ITS OWN headline geometry, with MST beside it. Their §4.2 1.3B setting:
# d_model=2048, 24 layers, d_thin=512, 15 blocks (1 shared softmax + 14 routed,
# top-3 = 4 active per token), d_ff,thin=2048. Published as 2.08B total / 0.61B
# active on FineWeb-Edu 20B tokens, 102 hours on 4xH200.
#
# The geometry is pinned with --model-dim regardless of the swept depth, like d16
# and d32, because 24 layers at d_model=2048 does not satisfy p08's aspect-ratio
# rule (24*64 = 1536, not 2048).
#
# ══ COST. READ THIS BEFORE LAUNCHING. ══
# Sized from estimate_flops() at this geometry (active training FLOPs):
#     M13_mol      8.77e19   21.2B tokens
#     M13_mst      2.34e19    6.3B tokens
#     M13_mst_k1   1.77e19    6.3B tokens
#     M13_dense    1.19e20   14.1B tokens   (commented out, see below)
# The three enabled arms total ~1.3e20, which is order 100-300 GPU-hours on
# H100/H200-class hardware. The dense arm alone would nearly double that, and it is
# NOT needed: the Pareto multiplier is computed by inverting the dense power-law fit
# (6.711 x^-0.1004), which is how every other number in this project was scored.
# Uncomment it only if a reviewer demands a same-sweep dense anchor at this scale.
#
# IF THAT IS TOO EXPENSIVE, USE `--group mol`, NOT A SMALLER TOKEN BUDGET. The same
# topology at L=8 costs 1/40th of this and is the conservative test rather than a
# weaker one, because MoL's compute disadvantage grows with scale (1.89x at L=8,
# 3.44x at L=16, 4.95x here). Cutting MoL's tokens instead would rig the comparison;
# see the `mol` group header for why.
#
# RUN ORDER MATTERS. The MST arms are ~4x cheaper than MoL and are the ones our
# claim rests on, so they come first: if they fail or the router collapses at this
# width, that is worth knowing before spending the MoL budget.
#
# ══ THESE ARE NOT COMPARABLE TO THEIR PUBLISHED TABLE 5. ══
# Deliberately, and it must be said in the paper. Our vocab is 65536 against their
# 50257, our lm_head is untied where theirs is tied, we carry value-embedding tables
# they do not, the data is ours not FineWeb-Edu, and the token budget is per-arm
# compute-optimal (10.5 x scaling params) rather than a fixed 20B for everyone. So
# our MoL prints ~2.56B total / ~1.18B active, not 2.08B / 0.61B. The comparison
# here is internal and controlled: MoL against MST under identical conditions.
# tests/test_mol.py is what pins fidelity to their published numbers, not this group.
#
# Set SEQ_LEN=4096 to match their T; the default 2048 keeps these on the same axis
# as the rest of p08. Their routed-block attention advantage grows with T, so 2048
# is the conservative choice for us and the unfavourable one for them.
#
# ALREADY KNOWN WITHOUT TRAINING, from estimate_flops() at this exact geometry:
# MoL costs 1.487e10 FLOPs/token against MST's 3.736e9 (4.0x), and 4.13e9 against
# 2.83e9 even after crediting MoL's routing and charging MST nothing for its own
# (1.46x). On training FLOPs MoL is 4.95x MST_k1. This group asks whether that
# 5x compute gap is bought back in bpb, which is the only way MoL wins.
if has m13; then
    echo ""; echo "### M13: MoL at its published 1.3B geometry, against MST"
    L13=24
    D13=2048
    DTHIN13=512
    GEOM13="--model-dim $D13 --head-dim 128"
    MST13="$(mst_config "$DTHIN13" 4)"
    BEST13="--mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 --mst-wo-mode dense"
    SEEDS_SAVE="$SEEDS"; SEEDS=1
    # Cheapest first, and the arm the paper rests on.
    run M13_mst_k1 "$L13" $GEOM13 $MST13 $BEST13 \
        --mst-stream-topk 1 --mst-stream-router-noise 1.0
    # Non-sparse MST, so the sparse delta is isolated at this scale too.
    run M13_mst    "$L13" $GEOM13 $MST13 $BEST13
    # Their headline topology, softmax in the routed blocks rather than Gated
    # DeltaNet (see OPEN_QUESTIONS.md Q1: their §5.3 dense-DeltaNet control matches
    # dense softmax within 0.01 PPL, but their Table 2 prices DeltaNet at 0.85 PPL
    # inside MoL, so this is their architecture and not their best number).
    run M13_mol    "$L13" $GEOM13 $(mol_config 15 1 3 "$DTHIN13")
    # Same-sweep dense anchor. Nearly doubles the group's cost; the fitted dense
    # curve makes it optional. Uncomment only if you need it in-sweep.
#    run M13_dense  "$L13" $GEOM13 --models base
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
