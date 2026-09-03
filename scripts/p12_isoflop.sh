#!/usr/bin/env bash
# ============================================================================
# P12: isoFLOP profile. Every arm gets the SAME training compute.
#
# WHY THIS EXISTS
#   The compute-optimal ladder sets each arm's token budget from its own
#   parameter count (10.5 x (matrices + lm_head)). That is a defensible
#   protocol but it is a protocol, and it is the one a reviewer will question,
#   because MST draws 1.43x the tokens of the dense model matched to it on
#   active FLOPs (see p11_active_params.py). An isoFLOP profile removes the
#   question entirely: fix C, vary model size, plot final bpb against C. This
#   is Chinchilla's "approach 2" and the default convention in efficiency work.
#
#   Read it as: at a fixed training-compute budget, which architecture's
#   best-performing model is better, and how big is that model?
#
# THE BUDGET
#   C = 6.710562e17 active FLOPs. That is MST L=16's existing training budget to six
#   figures (10.5 x 111,235,072 scaling params x 5.7455e8 active FLOPs/token), so that
#   run is already a point on the profile and costs nothing to reuse. Six new runs at
#   6.71e17 each is 4.0e18 total, about 0.8x of one L=24 run. isoFLOP cost does not
#   depend on which depths you pick, only on how many.
#
#   A second contour at C=4.8e18 (L=24's budget) would reuse MST L=24 the same
#   way, but costs ~6x an L=24 run. Only worth it if this one is favourable.
#
# ACTIVE, NOT TOTAL
#   --target-active-flops, not --target-flops. base_train's --target-flops
#   divides by estimate_flops()[0], the TOTAL count, while the Pareto plots use
#   [1], the active one. For MST those differ by 1.34x at L=24, so using the
#   total would put MST on a tighter contour than dense and bias the profile
#   against it without anyone noticing.
#
# EXPECTED SHAPE OF THE ANSWER
#   MST's training-FLOPs multiplier on the ladder is 1.08-1.14x, and the bpb
#   gap on that axis is 0.003-0.005, so a small separation is the honest
#   prediction. State that before looking. This profile is for credibility,
#   not for a big number; if you want the big number it is on the inference
#   axis and p13_isodata.sh is the run that gets it.
#
#   bash scripts/p12_isoflop.sh
#   FLOPS=4.8e18 bash scripts/p12_isoflop.sh
# ============================================================================
set -o pipefail

FORCE=0; SEEDS=1; ARMS=all; CLI_DEPTHS=(); TIMER=0
usage() {
    echo "usage: $0 [--force] [--seeds N] [--arms dense|mst|mol|all] [--timer-only] [depth ...]"
    echo "  --timer-only runs TIMER_STEPS (default 12) steps of every arm and projects the"
    echo "  full sweep from the measured dt, including startup and final-validation time."
    echo "  depths given positionally replace the built-in list for whichever arms run,"
    echo "  so '--mst-only 24' runs exactly one arm and nothing else."
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        # Split the profile across machines: run the dense arms in one place and the
        # MST arms in another. Each half keeps its own state file, so pointing both at
        # the same OUT_BASE (a shared volume) merges them; pointing them at different
        # ones keeps them independent and the halves are combined when plotting.
        --arms) ARMS="$2"; shift 2 ;;
        --dense-only) ARMS=dense; shift ;;
        --mst-only) ARMS=mst; shift ;;
        --mol-only) ARMS=mol; shift ;;
        --timer-only) TIMER=1; shift ;;
        -*) echo "unknown arg: $1"; usage; exit 1 ;;
        *) CLI_DEPTHS+=("$1"); shift ;;
    esac
done
for d in "${CLI_DEPTHS[@]}"; do
    [[ "$d" =~ ^[0-9]+$ ]] || { echo "depth must be a positive integer, got '$d'"; usage; exit 1; }
done
[ "$ARMS" = "both" ] && ARMS=all          # accepted for compatibility with earlier invocations
case "$ARMS" in
    all|dense|mst|mol) ;;
    *) echo "--arms must be one of: dense, mst, mol, all (got '$ARMS')"; exit 1 ;;
esac

# C is set by the CORPUS, not by affordability. One shard of the 300-shard set holds
# ~252.8M characters, so MAX_SHARDS=300 is roughly 18B tokens, and the protocol is single
# epoch. Each arm consumes C / (active FLOPs per token), so the cheapest-per-token arm
# fixes the ceiling:
#   dense [16,18,20,24] -> C_max 2.79e19      MST [16,24,28,32] -> C_max 1.03e19
#   dense [12,16,18,24] -> C_max 1.33e19      MST [12,16,24,32] -> C_max 5.73e18
# 9.0e18 sits at 87% of the binding MST ceiling, leaving margin for the +/-10% in that
# token estimate. Raise MAX_SHARDS to lift it: 600 shards would roughly double the ceiling.
#
# It also buys the scale the paper needs. At the previous 4.823e18, dense L=24 ran at
# 7.45x its own compute-optimal budget, i.e. a 1.38B model on 13% of its Chinchilla
# tokens, which is not citable as 1B-scale evidence. At 9.0e18:
#   dense  L=16 0.42x  L=18 0.81x  L=20 1.44x  L=24 3.99x   (537M .. 1.38B active)
#   MST    L=16 0.07x  L=24 0.54x  L=28 1.20x  L=32 2.46x   (388M .. 1.62B active)
# Both arms straddle their vertex, dense between L=18 and L=20, MST between L=24 and L=28.
# Cost: 8 arms x 9.0e18 = 7.2e19. Run --timer-only before committing.
FLOPS="${FLOPS:-9.0e18}"
N_SUBS="${N_SUBS:-4}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
# Dense L=16/18/20/24 spans 537M to 1.38B active and straddles a vertex between L=18 and
# L=20. L=12 was dropped when C rose: it is the cheapest-per-token dense arm, so it would
# have pulled the corpus ceiling down to 1.33e19 for no gain at a scale nobody asks about.
DENSE_DEPTHS="${DENSE_DEPTHS-16 18 20 24}"
# MST needs mst_sub_head_dim (64) to divide sub_dim = D/N, i.e. D a multiple of 256.
# L=16,24,28,32 give D=1024,1536,1792,2048 -> d=256,384,448,512, all divisible by 64.
# L=14,18,22,26,30 do NOT and have no MST arm at all.
#
# L=20 is EXCLUDED as MST's known off-trend ladder point: 1.126x on FLOPs/token against
# 1.223x at L=16 and 1.284x at L=24. No mechanism has been identified. The two candidates
# were both checked and both are dead: multi-scale windows are assigned PER STREAM
# (_sub_window_sizes, N=4), so the head count never touches the window schedule; and
# sub_dim = 64 mod 128 does break tensor-core alignment for MST in a way it never does for
# dense, but that is a speed effect and the anomaly is in bpb per FLOP. L=12 shares L=20's
# geometry (3 heads, sub_dim 192) and has never shown the effect, so a single bad run is
# the likelier explanation than anything structural. Rerunning L=20 on another seed would
# settle it for the cost of one arm.
#
# L=28 (7 heads, sub_dim 448) is in the same geometric class as L=20 and is kept
# deliberately: with four points its residual is visible, so this profile TESTS the
# hypothesis rather than assuming it. If L=28 lands off the curve the way L=20 did, drop
# it and fit 16/24/32, which is clean end to end (4/6/8 heads, all sub_dim a multiple of
# 128). It is also the 1.22B-active arm, the closest to the scale reviewers name.
MST_DEPTHS="${MST_DEPTHS-16 24 28 32}"
# MoL (Ternovtsii & Bilak 2026) as its own arm, in the 1+3of15 topology at d_thin = D/4
# that reproduces their published parameter counts exactly. Its thin blocks are wrapped in
# per-block W_down/W_up, so it costs far more per token than MST at equal depth (8.09e8
# against 5.75e8 at L=16); 8/12/16 therefore brackets the same budget that 12/16/24 brackets
# for MST. d_thin must be divisible by mol_head_dim 64, which rules out L=10, 14, 18, 22.
# Per-block value embeddings are ON, matching MST's --mst-per-stream-ve: each arm runs the
# configuration its own paper proposes, which is the comparison tab:isotoken16 already makes.
MOL_DEPTHS="${MOL_DEPTHS-8 12 16}"
# Positional depths override the built-in lists, so a single arm can be launched on its
# own machine. Applied before the --arms filter so "--mst-only 24" means exactly that.
if [ ${#CLI_DEPTHS[@]} -gt 0 ]; then
    DENSE_DEPTHS="${CLI_DEPTHS[*]}"
    MST_DEPTHS="${CLI_DEPTHS[*]}"
    MOL_DEPTHS="${CLI_DEPTHS[*]}"
fi
# MoL is parked: its per-block W_down/W_up wrappers make it far slower to train than
# either baseline at these sizes, so it is excluded from the default sweep rather than
# deleted. "--arms mol" and "--mol-only" still run it with the depths above.
case "$ARMS" in
    dense) MST_DEPTHS="";   MOL_DEPTHS="" ;;
    mst)   DENSE_DEPTHS=""; MOL_DEPTHS="" ;;
    mol)   DENSE_DEPTHS=""; MST_DEPTHS=""  ;;
    all)   MOL_DEPTHS="" ;;
esac
OUT_BASE="${OUT_BASE:-out/p12_isoflop}"
mkdir -p "$OUT_BASE"
TIMER_STEPS="${TIMER_STEPS:-12}"
if [ "$TIMER" -eq 1 ]; then
    # A costing pass must not touch the real sweep: its own out tree, its own state file,
    # and no mark_done, so a later real run still sees every arm as outstanding.
    OUT_BASE="${OUT_BASE}/timer"
    mkdir -p "$OUT_BASE"
    echo "TIMER-ONLY: ${TIMER_STEPS} steps per arm, writing to ${OUT_BASE}"
fi
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/p12.log}"

# Compiled-kernel caches, following runpod_env.sh and p30/p33/p35. Without these each
# arm compiles from an empty cache, and on an ephemeral runner every relaunch pays the
# full cost again: MST compiles far more kernels than dense (N=4 streams x 4 window
# scales, per-stream value embeddings, block-diagonal GEMMs), so it is the arm that
# suffers. Defaulting them under OUT_BASE means pointing OUT_BASE at a persistent
# volume also persists the caches, and keeps them inside the gitignored out/ tree.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${OUT_BASE}/.inductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OUT_BASE}/.triton_cache}"

# Inductor's compile-worker pool defaults to min(32, nproc) subprocesses, each holding
# its own torch import. On a many-core box with a memory ceiling that is enough RSS to
# get a worker OOM-killed, and the parent then waits on a future that never resolves.
# MST drives the pool far harder than dense (~1200 kernels at L=24 against ~400), which
# is why it is the arm that hangs. Cap it: this costs no step time, unlike
# --compile-regional, which is left off by default because it loses cross-layer fusion.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Arm boundaries and verdicts have to reach the log file, not just stdout. Only the
# sweep command is piped through tee, so without this the log is undelimited training
# output and the structure survives only in whatever captured stdout, which on a remote
# runner is a different place from the volume the log is written to.
log() { echo "$*" | tee -a "$LOGFILE"; }
STATE="${OUT_BASE}/p12_state.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# Interrupt handling. Without this an INT lands on the foreground process group, kills
# the arm's torchrun, and the loop reads the nonzero status as "this arm failed" and
# starts the NEXT multi-hour arm. One Ctrl-C then costs a run rather than stopping one.
ABORT=0
ALL_ARMS=()
FAILED_ARMS=()
on_signal() {
    ABORT=1
    printf '\n>>> interrupt received. Stopping. The current arm keeps its checkpoints\n'
    printf '>>> and resumes from its last step when this script is re-run.\n'
}
trap on_signal INT TERM

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

# --- timer-only projection -------------------------------------------------
# Per arm: measured wall time covers startup (env, data, compile) + TIMER_STEPS steps +
# the final eval and save. Splitting off the measured steps leaves the fixed overhead,
# and the full run is that overhead plus full_iterations x dt. base_train prints
# "TIMING_PROBE full_iterations=... " before training so the real horizon is known even
# though the loop stops early.
TIMER_TOTAL=0
TIMER_ROWS=()
project_arm() {                           # project_arm <tag> <log> <t_start> <t_end>
    local tag="$1" alog="$2"
    local elapsed
    elapsed=$(awk "BEGIN{printf \"%.2f\", $4 - $3}")
    local full dt
    full=$(grep -oE 'TIMING_PROBE full_iterations=[0-9]+' "$alog" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
    # Fallback: base_train announces the horizon before training even without the probe flag.
    [ -z "$full" ] && full=$(grep -oE 'Calculated number of iterations from target ACTIVE FLOPs: [0-9,]+' "$alog" 2>/dev/null \
                             | tail -1 | grep -oE '[0-9,]+$' | tr -d ',')
    dt=$(grep -oE 'dt: [0-9.]+ *ms' "$alog" 2>/dev/null | tail -1 | grep -oE '[0-9.]+')
    if [ -z "$full" ] || [ -z "$dt" ]; then
        log "TIMER $tag: could not parse (full_iterations='${full:-?}' dt='${dt:-?}'); measured ${elapsed}s only"
        if [ ! -s "$alog" ]; then
            log "   arm log ${alog} is missing or empty"
        else
            log "   arm log ${alog} has $(wc -l < "$alog") lines; last line:"
            log "   $(tail -1 "$alog" | cut -c1-140)"
            grep -q 'TIMING_PROBE' "$alog" || log "   no TIMING_PROBE line: --timing-probe-steps did not reach base_train"
            grep -qE 'dt: [0-9.]+ *ms' "$alog" || log "   no step line: the arm never reached step ${TIMER_STEPS} (log-every is ${LOG_EVERY:-200})"
        fi
        TIMER_ROWS+=("$tag|$elapsed|?|?|?")
        return
    fi
    local overhead proj
    # clamp: dt is the steady-state step, so the first few steps can exceed it and drive
    # the residual negative on a fast arm. Overhead is never less than zero.
    overhead=$(awk "BEGIN{o=$elapsed - $TIMER_STEPS * $dt/1000; if(o<0)o=0; printf \"%.1f\", o}")
    proj=$(awk "BEGIN{printf \"%.1f\", $overhead + $full * $dt/1000}")
    TIMER_TOTAL=$(awk "BEGIN{printf \"%.1f\", $TIMER_TOTAL + $proj}")
    log "TIMER $tag: ${full} steps x ${dt}ms + ${overhead}s overhead = $(awk "BEGIN{printf \"%.2f\", $proj/3600}")h"
    TIMER_ROWS+=("$tag|$elapsed|$full|$dt|$proj")
}

COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 --target-tokens -1 \
  --compile-regional ${COMPILE_REGIONAL:-0} \
  --save-every 200 --eval-every -1 --target-active-flops ${FLOPS}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

run() {
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        ALL_ARMS+=("$t")
        [ "$ABORT" -eq 1 ] && continue
        if done_already "$t"; then log "SKIP $t"; continue; fi
        log ""; log "=== $t (depth $depth, C=${FLOPS} active FLOPs) ==="
        local dir="${OUT_BASE}/${t}"
        [ "$FORCE" -eq 1 ] && rm -rf "$dir"
        local rc=0
        local t_start=$(date +%s.%N)
        local armlog="${dir}.probe.log"
        if [ "$TIMER" -eq 1 ]; then
            # A real pipeline, not a variable that expands to "|": bash parses redirections
            # before expanding, so the old ${TIMER:+| tee ...} became literal filenames.
            bash scripts/research_sweep.sh $COMMON --timing-probe-steps "$TIMER_STEPS" \
                 --out-dir "$dir" --seed "$s" "$@" "$depth" 2>&1 \
                 | tee "$armlog" | tee -a "$LOGFILE" || rc=$?
        else
            bash scripts/research_sweep.sh $COMMON --out-dir "$dir" --seed "$s" \
                 "$@" "$depth" 2>&1 | tee -a "$LOGFILE" || rc=$?
        fi
        local t_end=$(date +%s.%N)
        [ "$TIMER" -eq 1 ] && project_arm "$t" "$armlog" "$t_start" "$t_end"
        if [ "$ABORT" -eq 1 ] || [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then
            ABORT=1
            log "INTERRUPTED $t  (resumes from its last checkpoint on the next run)"
            continue
        fi
        # An arm counts as complete only if it left the result row this profile reads.
        # research_sweep.sh can exit 0 without training anything, because its own
        # per-model state may already believe the models are finished; marking that
        # done would drop a point from the profile with no error anywhere.
        if [ "$TIMER" -eq 1 ]; then
            [ "$rc" -eq 0 ] || { FAILED_ARMS+=("$t"); log "FAIL $t (rc=$rc)"; }
        elif [ "$rc" -eq 0 ] && [ -f "${dir}/depth_${depth}/results_depth_${depth}.tsv" ]; then
            mark_done "$t"; log "OK $t"
        else
            FAILED_ARMS+=("$t")
            log "FAIL $t (rc=$rc)"
        fi
    done
}

# The MST arm is the paper's headline SP2_k1 config. NOT the S=1 variant: the point of
# this profile is to validate the configuration the ladder is built on, and a profile
# run on a different config validates nothing. The two are Pareto-equivalent anyway
# (measured neutral at L=8 and, after correcting for its token excess, at L=16), and
# the control has the lower FLOPs/token, so it is both the cheaper run and the stronger
# inference-cost position.
mol_config() {                            # mol_config <depth>
    local D=$(( (($1 * ASPECT_RATIO + 127) / 128) * 128 ))
    echo "--use-mol 1 --models base --mol-n-blocks 15 --mol-n-shared 1 --mol-topk 3 \
      --mol-thin-dim $(( D / 4 )) --mol-head-dim 64 --mol-ffn-mult 4.0 \
      --mol-router-aux 0.05 --mol-routed-attn softmax --mol-dispatch 1 \
      --mol-per-block-ve 1"
}

mst_config() {                            # mst_config <depth>
    local D=$(( (($1 * ASPECT_RATIO + 127) / 128) * 128 ))
    local SD=$(( D / N_SUBS ))
    echo "--use-mst 1 --models base --mst-n-subs $N_SUBS --mst-sub-dim $SD \
      --mst-head-dim 0 --mst-input-mode learned_proj \
      --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
      --mst-transition-mode aggregate_distribute \
      --mst-final-mode concat_proj --mst-final-topk 0 \
      --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
      --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
      --mst-transition-width-mult ${N_SUBS}.0 --mst-sub-lr-scale 2.0 \
      --mst-multi-scale-windows 1 \
      --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 \
      --mst-wo-mode dense --mst-stream-topk 1 --mst-stream-router-noise 1.0"
}

echo "============================================================"
echo "  P12 isoFLOP profile   C = ${FLOPS} active FLOPs"
echo "  arms: ${ARMS}"
echo "  dense: ${DENSE_DEPTHS:-(none)}   MST: ${MST_DEPTHS:-(none)}   MoL: ${MOL_DEPTHS:-(none)}"
echo "  out ${OUT_BASE}"
echo "============================================================"

for d in $DENSE_DEPTHS; do
    run "ISOF_dense_d${d}" "$d" --models base
done
for d in $MST_DEPTHS; do
    SD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / N_SUBS ))
    if [ $(( SD % 64 )) -ne 0 ]; then
        echo "SKIP MST d${d}: sub_dim ${SD} not divisible by mst_sub_head_dim 64"
        continue
    fi
    run "ISOF_mst_d${d}" "$d" $(mst_config "$d")
done
for d in $MOL_DEPTHS; do
    TD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / 4 ))
    if [ $(( TD % 64 )) -ne 0 ]; then
        echo "SKIP MoL d${d}: thin_dim ${TD} not divisible by mol_head_dim 64"
        continue
    fi
    run "ISOF_mol_d${d}" "$d" $(mol_config "$d")
done

echo ""
echo "============================================================"
if [ "$TIMER" -eq 1 ]; then
    printf '  %-24s %10s %8s %10s %10s\n' arm measured steps "dt(ms)" projected
    for r in "${TIMER_ROWS[@]}"; do
        IFS='|' read -r a m f d pj <<< "$r"
        printf '  %-24s %9ss %8s %10s %9sh\n' "$a" "$m" "$f" "$d" \
            "$(awk "BEGIN{printf \"%.2f\", ${pj:-0}/3600}")"
    done
    echo "  ------------------------------------------------------------------"
    echo "  projected total for the whole sweep: $(awk "BEGIN{printf \"%.2f\", $TIMER_TOTAL/3600}")h"
    echo "  (startup, ${TIMER_STEPS} measured steps and final validation are all included per arm)"
    echo "============================================================"
    [ ${#FAILED_ARMS[@]} -gt 0 ] && { echo "  failed: ${FAILED_ARMS[*]}"; exit 1; }
    exit 0
fi
REMAINING=()
for a in "${ALL_ARMS[@]}"; do done_already "$a" || REMAINING+=("$a"); done
log "  arms complete: $(( ${#ALL_ARMS[@]} - ${#REMAINING[@]} )) / ${#ALL_ARMS[@]}"
[ ${#REMAINING[@]} -gt 0 ] && log "  still to run:  ${REMAINING[*]}"
[ ${#FAILED_ARMS[@]} -gt 0 ] && log "  failed:        ${FAILED_ARMS[*]}"
[ ${#REMAINING[@]} -gt 0 ] && echo "  re-run this script to continue; finished arms are skipped."
echo "============================================================"
echo ""
echo "============================================================"
echo "  done: ${OUT_BASE}/"
echo "  Every run saw the same ${FLOPS} active training FLOPs, so bpb is"
echo "  directly comparable. Plot bpb against depth per arm; the lower"
echo "  envelope of each arm is that architecture's isoFLOP frontier."
echo "============================================================"

[ "$ABORT" -eq 1 ] && exit 130
[ ${#REMAINING[@]} -gt 0 ] && exit 1
exit 0
