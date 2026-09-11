#!/usr/bin/env bash
# ============================================================================
# P16: isoFLOP comparison — MST Headline (Top-1 Sparse) vs. Shared D->D->D FFN
#
# WHY THIS EXISTS
#   Investigate whether replacing MST's 4 parallel sub-FFNs + top-1 sparse routing
#   with a single shared D -> D -> D FFN (dense cross-sub refinement for all subs)
#   yields a better quality-compute trade-off under a fixed training budget.
#
# THE BUDGET
#   Training compute is derived using the repo's Chinchilla scaling law:
#     tokens = 10.5 * (transformer_matrices + lm_head)
#     training_flops = tokens * active_flops_per_token
#   
#   Evaluated at Depth 8 (D=512, sub_dim=128):
#     - MST SP2_k1:     26,760,192 scaling params * 10.5 = 280.98M tok * 1.5622e8 = 4.3895e+16 FLOPs
#     - MST Shared D:   26,743,808 scaling params * 10.5 = 280.81M tok * 1.7500e8 = 4.9141e+16 FLOPs
#   The higher of the two is 4.914076e+16 (Shared D->D), which is fixed as the
#   training compute budget for BOTH arms at depth 8.
#
#   (For Depth 12: max is 2.334370e+17; for Depth 16: max is 8.464552e+17).
#
# USAGE
#   bash scripts/p16_isoflop_shared_d.sh              # Runs Depth 8 at C = 4.914076e+16
#   bash scripts/p16_isoflop_shared_d.sh --timer-only  # Quick 20-step timing probe
#   bash scripts/p16_isoflop_shared_d.sh --arms shared_d
#   bash scripts/p16_isoflop_shared_d.sh 12            # Runs Depth 12 at C = 2.334370e+17
# ============================================================================
set -o pipefail

FORCE=0; SEEDS=1; ARMS=all; CLI_DEPTHS=(); TIMER=0
usage() {
    echo "usage: $0 [--force] [--seeds N] [--arms mst|shared_d|sandwiched|grouped_shared|swiglu|new|dense|all] [--timer-only] [depth ...]"
    echo "  --new-only runs all 3 new arms (sandwiched, grouped_shared, swiglu) together."
    echo "  --timer-only runs TIMER_STEPS (default 20) steps of every arm and projects the"
    echo "  full sweep from the measured dt, including startup and final-validation time."
    echo "  depths given positionally replace the built-in list for whichever arms run."
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --mst-only) ARMS=mst; shift ;;
        --shared-d-only) ARMS=shared_d; shift ;;
        --sandwiched-only) ARMS=sandwiched; shift ;;
        --grouped-shared-only) ARMS=grouped_shared; shift ;;
        --swiglu-only) ARMS=swiglu; shift ;;
        --new-only|--new-arms-only|--new-arms) ARMS=new; shift ;;
        --dense-only) ARMS=dense; shift ;;
        --timer-only) TIMER=1; shift ;;
        -*) echo "unknown arg: $1"; usage; exit 1 ;;
        *) CLI_DEPTHS+=("$1"); shift ;;
    esac
done
for d in "${CLI_DEPTHS[@]}"; do
    [[ "$d" =~ ^[0-9]+$ ]] || { echo "depth must be a positive integer, got '$d'"; usage; exit 1; }
done
case "$ARMS" in
    all|mst|shared_d|sandwiched|grouped_shared|swiglu|new|dense) ;;
    *) echo "--arms must be one of: mst, shared_d, sandwiched, grouped_shared, swiglu, new, dense, all (got '$ARMS')"; exit 1 ;;
esac

N_SUBS="${N_SUBS:-4}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"

# Default depth is L=8 (D=512, d=128), the standard exploration & parity depth.
# Override via positional CLI args (e.g. `bash scripts/p16_isoflop_shared_d.sh 12`)
MST_DEPTHS="${MST_DEPTHS-8}"
SHARED_D_DEPTHS="${SHARED_D_DEPTHS-8}"
SANDWICHED_DEPTHS="${SANDWICHED_DEPTHS-8}"
GROUPED_SHARED_DEPTHS="${GROUPED_SHARED_DEPTHS-8}"
SWIGLU_DEPTHS="${SWIGLU_DEPTHS-8}"
DENSE_DEPTHS="${DENSE_DEPTHS-8}"

if [ ${#CLI_DEPTHS[@]} -gt 0 ]; then
    MST_DEPTHS="${CLI_DEPTHS[*]}"
    SHARED_D_DEPTHS="${CLI_DEPTHS[*]}"
    SANDWICHED_DEPTHS="${CLI_DEPTHS[*]}"
    GROUPED_SHARED_DEPTHS="${CLI_DEPTHS[*]}"
    SWIGLU_DEPTHS="${CLI_DEPTHS[*]}"
    DENSE_DEPTHS="${CLI_DEPTHS[*]}"
fi

# Function to return the highest Chinchilla training FLOPs between the two arms for any depth
default_flops_for_depth() {
    case "$1" in
        8)  echo "4.914076e+16" ;;
        12) echo "2.334370e+17" ;;
        16) echo "8.464552e+17" ;;
        24) echo "6.445339e+18" ;;
        *)  echo "4.914076e+16" ;;
    esac
}

PRIMARY_DEPTH=$(echo "$MST_DEPTHS $SHARED_D_DEPTHS $SANDWICHED_DEPTHS $GROUPED_SHARED_DEPTHS $SWIGLU_DEPTHS $DENSE_DEPTHS" | tr ' ' '\n' | grep -v '^$' | head -1)
PRIMARY_DEPTH="${PRIMARY_DEPTH:-8}"
FLOPS="${FLOPS:-$(default_flops_for_depth "$PRIMARY_DEPTH")}"

case "$ARMS" in
    mst)            SHARED_D_DEPTHS=""; SANDWICHED_DEPTHS=""; GROUPED_SHARED_DEPTHS=""; SWIGLU_DEPTHS=""; DENSE_DEPTHS="" ;;
    shared_d)       MST_DEPTHS="";      SANDWICHED_DEPTHS=""; GROUPED_SHARED_DEPTHS=""; SWIGLU_DEPTHS=""; DENSE_DEPTHS="" ;;
    sandwiched)     MST_DEPTHS="";      SHARED_D_DEPTHS="";   GROUPED_SHARED_DEPTHS=""; SWIGLU_DEPTHS=""; DENSE_DEPTHS="" ;;
    grouped_shared) MST_DEPTHS="";      SHARED_D_DEPTHS="";   SANDWICHED_DEPTHS="";     SWIGLU_DEPTHS=""; DENSE_DEPTHS="" ;;
    swiglu)         MST_DEPTHS="";      SHARED_D_DEPTHS="";   SANDWICHED_DEPTHS="";     GROUPED_SHARED_DEPTHS=""; DENSE_DEPTHS="" ;;
    new)            MST_DEPTHS="";      SHARED_D_DEPTHS="";   DENSE_DEPTHS="" ;;  # Runs all 3 new arms alone
    dense)          MST_DEPTHS="";      SHARED_D_DEPTHS="";   SANDWICHED_DEPTHS="";     GROUPED_SHARED_DEPTHS=""; SWIGLU_DEPTHS="" ;;
    all)            DENSE_DEPTHS="" ;;   # Focus comparison on all MST variants
esac

OUT_BASE="${OUT_BASE:-out/p16_isoflop_shared_d}"
mkdir -p "$OUT_BASE"
TIMER_STEPS="${TIMER_STEPS:-20}"
if [ "$TIMER" -eq 1 ]; then
    OUT_BASE="${OUT_BASE}/timer"
    mkdir -p "$OUT_BASE"
    echo "TIMER-ONLY: ${TIMER_STEPS} steps per arm, writing to ${OUT_BASE}"
fi
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/p16.log}"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${OUT_BASE}/.inductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OUT_BASE}/.triton_cache}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

log() { echo "$*" | tee -a "$LOGFILE"; }
STATE="${OUT_BASE}/p16_state.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

ABORT=0
ALL_ARMS=()
FAILED_ARMS=()
on_signal() {
    ABORT=1
    printf '\n>>> interrupt received. Stopping.\n'
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

TIMER_TOTAL=0
TIMER_ROWS=()
project_arm() {
    local tag="$1" alog="$2"
    local elapsed
    elapsed=$(awk "BEGIN{printf \"%.2f\", $4 - $3}")
    local res
    res=$(grep -oE 'TIMING_PROBE_RESULT [^|]*' "$alog" 2>/dev/null | tail -1)
    local full dt timed
    if [ -n "$res" ]; then
        dt=$(printf '%s' "$res"   | grep -oE 'dt_ms=[0-9.]+'          | cut -d= -f2)
        timed=$(printf '%s' "$res"| grep -oE 'timed_steps=[0-9]+'     | cut -d= -f2)
        full=$(printf '%s' "$res" | grep -oE 'full_iterations=[0-9]+' | cut -d= -f2)
        [ "${timed:-0}" -lt 1 ] && dt=""
    fi
    [ -z "$full" ] && full=$(grep -oE 'TIMING_PROBE full_iterations=[0-9]+' "$alog" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
    [ -z "$full" ] && full=$(grep -oE 'Calculated number of iterations from target ACTIVE FLOPs: [0-9,]+' "$alog" 2>/dev/null \
                             | tail -1 | grep -oE '[0-9,]+$' | tr -d ',')
    [ -z "$dt" ] && dt=$(grep -oE 'dt: [0-9.]+ *ms' "$alog" 2>/dev/null | tail -1 | grep -oE '[0-9.]+')
    if [ -z "$full" ] || [ -z "$dt" ]; then
        log "TIMER $tag: could not parse (full_iterations='${full:-?}' dt='${dt:-?}'); measured ${elapsed}s only"
        TIMER_ROWS+=("$tag|?|?|?|?")
        return
    fi
    local overhead proj
    overhead=$(awk "BEGIN{o=$elapsed - $TIMER_STEPS * $dt/1000; if(o<0)o=0; printf \"%.1f\", o}")
    proj=$(awk "BEGIN{printf \"%.1f\", $overhead + $full * $dt/1000}")
    TIMER_TOTAL=$(awk "BEGIN{printf \"%.1f\", $TIMER_TOTAL + $proj}")
    log "TIMER $tag: ${full} steps x ${dt}ms + ${overhead}s overhead = $(awk "BEGIN{printf \"%.2f\", $proj/3600}")h"
    TIMER_ROWS+=("$tag|$overhead|$full|$dt|$proj")
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
        [ "$TIMER" -eq 1 ] && rm -rf "$dir"
        local rc=0
        local t_start=$(date +%s.%N)
        local armlog="${dir}.probe.log"
        if [ "$TIMER" -eq 1 ]; then
            bash scripts/research_sweep.sh $COMMON --timing-probe-steps "$TIMER_STEPS" \
                 --out-dir "$dir" --seed "$s" "$@" "$depth" 2>&1 \
                 | tee "$armlog" >> "$LOGFILE" || rc=$?
        else
            bash scripts/research_sweep.sh $COMMON --out-dir "$dir" --seed "$s" \
                 "$@" "$depth" 2>&1 | tee -a "$LOGFILE" || rc=$?
        fi
        local t_end=$(date +%s.%N)
        [ "$TIMER" -eq 1 ] && project_arm "$t" "$armlog" "$t_start" "$t_end"
        if [ "$ABORT" -eq 1 ] || [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then
            ABORT=1
            log "INTERRUPTED $t"
            continue
        fi
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

# Arm 1: The paper's headline MST SP2_k1 arm (Batched MST with top-1 FFN routing)
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

# Arm 2: Proposed Shared D->d->D FFN arm (dense cross-sub FFN refinement for all subs)
mst_shared_d_config() {                   # mst_shared_d_config <depth>
    local D=$(( (($1 * ASPECT_RATIO + 127) / 128) * 128 ))
    local SD=$(( D / N_SUBS ))
    local INNER="${FFN_INNER_DIM:-$SD}"
    echo "--use-mst 1 --models base --mst-n-subs $N_SUBS --mst-sub-dim $SD \
      --mst-head-dim 0 --mst-input-mode learned_proj \
      --mst-routing-mode soft_weighted --mst-routing-topk 0 \
      --mst-ffn-mode shared_dense --mst-ffn-inner-dim $INNER \
      --mst-transition-mode aggregate_distribute \
      --mst-final-mode concat_proj --mst-final-topk 0 \
      --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
      --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
      --mst-transition-width-mult ${N_SUBS}.0 --mst-sub-lr-scale 2.0 \
      --mst-multi-scale-windows 1 \
      --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 \
      --mst-wo-mode dense --mst-stream-topk 0"
}

# Arm 3: Sandwiched / Alternating Shared FFN (every 2 layers, with wider M=D)
mst_sandwiched_config() {                 # mst_sandwiched_config <depth>
    local D=$(( (($1 * ASPECT_RATIO + 127) / 128) * 128 ))
    local SD=$(( D / N_SUBS ))
    local INNER="${FFN_INNER_DIM:-$D}"
    echo "--use-mst 1 --models base --mst-n-subs $N_SUBS --mst-sub-dim $SD \
      --mst-head-dim 0 --mst-input-mode learned_proj \
      --mst-routing-mode soft_weighted --mst-routing-topk 0 \
      --mst-ffn-mode shared_dense --mst-ffn-every 2 --mst-ffn-inner-dim $INNER \
      --mst-transition-mode aggregate_distribute \
      --mst-final-mode concat_proj --mst-final-topk 0 \
      --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
      --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
      --mst-transition-width-mult ${N_SUBS}.0 --mst-sub-lr-scale 2.0 \
      --mst-multi-scale-windows 1 \
      --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 \
      --mst-wo-mode dense --mst-stream-topk 0"
}

# Arm 4: Grouped Up-Projection + Shared Down-Projection (private stream expansion, joint mixing)
mst_grouped_shared_config() {             # mst_grouped_shared_config <depth>
    local D=$(( (($1 * ASPECT_RATIO + 127) / 128) * 128 ))
    local SD=$(( D / N_SUBS ))
    local INNER="${FFN_INNER_DIM:-$SD}"
    echo "--use-mst 1 --models base --mst-n-subs $N_SUBS --mst-sub-dim $SD \
      --mst-head-dim 0 --mst-input-mode learned_proj \
      --mst-routing-mode soft_weighted --mst-routing-topk 0 \
      --mst-ffn-mode grouped_up_shared_down --mst-ffn-inner-dim $INNER \
      --mst-transition-mode aggregate_distribute \
      --mst-final-mode concat_proj --mst-final-topk 0 \
      --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
      --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
      --mst-transition-width-mult ${N_SUBS}.0 --mst-sub-lr-scale 2.0 \
      --mst-multi-scale-windows 1 \
      --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 \
      --mst-wo-mode dense --mst-stream-topk 0"
}

# Arm 5: Shared SwiGLU FFN (multiplicative gating)
mst_swiglu_config() {                     # mst_swiglu_config <depth>
    local D=$(( (($1 * ASPECT_RATIO + 127) / 128) * 128 ))
    local SD=$(( D / N_SUBS ))
    local INNER="${FFN_INNER_DIM:-$SD}"
    echo "--use-mst 1 --models base --mst-n-subs $N_SUBS --mst-sub-dim $SD \
      --mst-head-dim 0 --mst-input-mode learned_proj \
      --mst-routing-mode soft_weighted --mst-routing-topk 0 \
      --mst-ffn-mode shared_swiglu --mst-ffn-inner-dim $INNER \
      --mst-transition-mode aggregate_distribute \
      --mst-final-mode concat_proj --mst-final-topk 0 \
      --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
      --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
      --mst-transition-width-mult ${N_SUBS}.0 --mst-sub-lr-scale 2.0 \
      --mst-multi-scale-windows 1 \
      --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 \
      --mst-wo-mode dense --mst-stream-topk 0"
}

echo "============================================================"
echo "  P16 isoFLOP comparison   C = ${FLOPS} active FLOPs"
echo "  arms: ${ARMS}"
echo "  out ${OUT_BASE}"
echo "============================================================"

for d in $MST_DEPTHS; do
    SD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / N_SUBS ))
    if [ $(( SD % 64 )) -ne 0 ]; then
        echo "SKIP MST d${d}: sub_dim ${SD} not divisible by 64"
        continue
    fi
    run "ISOF_mst_d${d}" "$d" $(mst_config "$d")
done

for d in $SHARED_D_DEPTHS; do
    SD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / N_SUBS ))
    if [ $(( SD % 64 )) -ne 0 ]; then
        echo "SKIP Shared-D d${d}: sub_dim ${SD} not divisible by 64"
        continue
    fi
    run "ISOF_mst_shared_d_d${d}" "$d" $(mst_shared_d_config "$d")
done

for d in $SANDWICHED_DEPTHS; do
    SD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / N_SUBS ))
    if [ $(( SD % 64 )) -ne 0 ]; then
        echo "SKIP Sandwiched d${d}: sub_dim ${SD} not divisible by 64"
        continue
    fi
    run "ISOF_mst_sandwiched_d${d}" "$d" $(mst_sandwiched_config "$d")
done

for d in $GROUPED_SHARED_DEPTHS; do
    SD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / N_SUBS ))
    if [ $(( SD % 64 )) -ne 0 ]; then
        echo "SKIP Grouped-Shared d${d}: sub_dim ${SD} not divisible by 64"
        continue
    fi
    run "ISOF_mst_grouped_shared_d${d}" "$d" $(mst_grouped_shared_config "$d")
done

for d in $SWIGLU_DEPTHS; do
    SD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / N_SUBS ))
    if [ $(( SD % 64 )) -ne 0 ]; then
        echo "SKIP SwiGLU d${d}: sub_dim ${SD} not divisible by 64"
        continue
    fi
    run "ISOF_mst_swiglu_d${d}" "$d" $(mst_swiglu_config "$d")
done

for d in $DENSE_DEPTHS; do
    run "ISOF_dense_d${d}" "$d" --models base
done

echo ""
echo "============================================================"
if [ "$TIMER" -eq 1 ]; then
    printf '  %-28s %11s %8s %10s %10s\n' arm "startup(s)" steps "dt(ms)" projected
    for r in "${TIMER_ROWS[@]}"; do
        IFS='|' read -r a m f d pj <<< "$r"
        printf '  %-28s %9ss %8s %10s %9sh\n' "$a" "$m" "$f" "$d" \
            "$(awk "BEGIN{printf \"%.2f\", ${pj:-0}/3600}")"
    done
    echo "  ------------------------------------------------------------------"
    echo "  projected total for the whole sweep: $(awk "BEGIN{printf \"%.2f\", $TIMER_TOTAL/3600}")h"
    echo "============================================================"
    [ ${#FAILED_ARMS[@]} -gt 0 ] && { echo "  failed: ${FAILED_ARMS[*]}"; exit 1; }
    exit 0
fi

REMAINING=()
for a in "${ALL_ARMS[@]}"; do done_already "$a" || REMAINING+=("$a"); done
log "  arms complete: $(( ${#ALL_ARMS[@]} - ${#REMAINING[@]} )) / ${#ALL_ARMS[@]}"
[ ${#REMAINING[@]} -gt 0 ] && log "  still to run:  ${REMAINING[*]}"
[ ${#FAILED_ARMS[@]} -gt 0 ] && log "  failed:        ${FAILED_ARMS[*]}"
echo "============================================================"
echo "  done: ${OUT_BASE}/"
echo "  Every run saw the same ${FLOPS} active training FLOPs."
echo "============================================================"

[ "$ABORT" -eq 1 ] && exit 130
[ ${#REMAINING[@]} -gt 0 ] && exit 1
exit 0
