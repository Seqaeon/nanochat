#!/usr/bin/env bash
# ============================================================================
# P03 MST Sweep — Stage 3: Transition Residual + Normalization
# ============================================================================
# Based on P02 Stage 2 results. Key finding:
#   - FFA + concat_proj_final is the dominant combo (1.11 val_bpb)
#   - aggdist + concat_proj_final is 2nd best (1.158)
#   - Transition had NO residual connection or pre-norm — now fixed
#
# Stage 3 experiments:
#   Group A: Re-test S2 top configs with transition residual + pre-norm
#   Group B: Head dimension exploration (head_dim=64)
#
# Key code change: MSTLayer.forward now wraps non-parallel transitions as:
#   normed = norm(sub_output)
#   transitioned = transition(normed)
#   output = sub_output + transitioned   ← residual around transition
#
# Usage:
#   bash scripts/p03_mst_sweep.sh [DEPTH]
#   bash scripts/p03_mst_sweep.sh --force [DEPTH]
#
# Default depth: 8
# ============================================================================

set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────────
FORCE=0
DEPTH=8
for arg in "$@"; do
    case $arg in
        --force) FORCE=1 ;;
        *)       DEPTH=$arg ;;
    esac
done

# ── Output directory ─────────────────────────────────────────────────────────
MST_OUT_BASE="${MST_OUT_BASE:-out/p03_mst_sweep}"
LOGFILE="${SWEEP_LOG:-${MST_OUT_BASE}/sweep_mst_p03_d${DEPTH}.log}"
STATE_FILE="${MST_OUT_BASE}/sweep_state_d${DEPTH}.json"
mkdir -p "$MST_OUT_BASE"

echo "═══════════════════════════════════════════════════════════════"
echo "  P03 MST Sweep — Stage 3: Transition Residual — Depth ${DEPTH}"
echo "  Output:  ${MST_OUT_BASE}"
echo "  State:   ${STATE_FILE}"
echo "  Log:     ${LOGFILE}"
echo "═══════════════════════════════════════════════════════════════"

# ── State management (JSON) ──────────────────────────────────────────────────
init_state() {
    if [ ! -f "$STATE_FILE" ]; then
        echo '{"completed":{},"started":{}}' > "$STATE_FILE"
    fi
}

check_completed() {
    local tag="$1"
    [ "$FORCE" -eq 1 ] && return 1
    python3 -c "
import json, sys
state = json.load(open('$STATE_FILE'))
sys.exit(0 if '$tag' in state.get('completed', {}) else 1)
" 2>/dev/null
}

mark_started() {
    local tag="$1"
    local run_dir="$2"
    python3 -c "
import json, datetime
state = json.load(open('$STATE_FILE'))
state.setdefault('started', {})['$tag'] = {
    'run_dir': '$run_dir',
    'started_at': datetime.datetime.now().isoformat()
}
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
"
}

mark_completed() {
    local tag="$1"
    python3 -c "
import json, datetime
state = json.load(open('$STATE_FILE'))
state.setdefault('completed', {})['$tag'] = {
    'completed_at': datetime.datetime.now().isoformat()
}
state.get('started', {}).pop('$tag', None)
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
"
}

print_header() {
    local id="$1" tag="$2" desc="$3"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$id] $tag"
    echo "  $desc"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

init_state

# ── Common flags ─────────────────────────────────────────────────────────────
MST_COMMON="--models base \
  --device-batch-size 128 --total-batch-size -1 --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 \
  --warmdown-ratio 0.20 \
  --final-lr-frac 0.05 \
  --research-dim -1 \
  --target-tokens 0 \
  --target-active-params 0 \
  --save-every 200 \
  --eval-every -1 \
  --use-mst 1 \
  --mst-n-subs 8 --mst-head-dim 0 \
  --mst-sub-dim 64"

# Add optional env-based flags
[ -n "${DATA_DIR:-}" ]      && MST_COMMON="$MST_COMMON --data-dir $DATA_DIR"
[ -n "${TOKENIZER_DIR:-}" ] && MST_COMMON="$MST_COMMON --tokenizer-dir $TOKENIZER_DIR"
[ -n "${MAX_SHARDS:-}" ]    && MST_COMMON="$MST_COMMON --max-shards $MAX_SHARDS"
[ "${USE_FP8:-0}" = "1" ]   && MST_COMMON="$MST_COMMON --fp8"

run_experiment() {
    local tag="$1"
    shift
    local desc="$1"
    shift

    if check_completed "$tag"; then
        echo "⏭  Skipping $tag (already completed)"
        return 0
    fi

    print_header "$tag" "$tag" "$desc"
    local run_dir="${MST_OUT_BASE}/${tag}"
    mark_started "$tag" "$run_dir"

    if bash scripts/research_sweep.sh $MST_COMMON \
      --out-dir "$run_dir" \
      "$@" \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $tag done"
        mark_completed "$tag"
    else
        echo "❌  $tag FAILED — will retry next run"
    fi
}

# ============================================================================
# Group A: Re-test S2 winners with transition residual + pre-norm
# ============================================================================
# S2 reference results (no residual):
#   learned+FFA+concat_proj      → 1.110 val_bpb  (row 26/28)
#   fixed+FFA+concat_proj        → 1.148 val_bpb  (row 27)
#   learned+aggdist+concat_proj  → 1.158 val_bpb  (row 25)
#   learned+FFA+aggregate_proj   → 1.194 val_bpb  (row 21)

# A1: Best S2 config + transition residual (S2 baseline: 1.110)
run_experiment "S3A1_FFA_CONCAT_RESID_D${DEPTH}" \
    "Stage 3: learned_proj + FFA(soft) + concat_proj [+transition residual+norm]" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj \
    --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 \
    --mst-diversity-weight 0.0

# A2: Aggdist + concat_proj + transition residual (S2 baseline: 1.158)
run_experiment "S3A2_AGGDIST_CONCAT_RESID_D${DEPTH}" \
    "Stage 3: learned_proj + aggdist + concat_proj [+transition residual+norm]" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj \
    --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 \
    --mst-diversity-weight 0.0

# A3: Fixed slice + FFA + concat_proj + transition residual (S2 baseline: 1.148)
run_experiment "S3A3_FIXED_FFA_CONCAT_RESID_D${DEPTH}" \
    "Stage 3: fixed_slice + FFA(soft) + concat_proj [+transition residual+norm]" \
    --mst-input-mode fixed_slice \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj \
    --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 \
    --mst-diversity-weight 0.0

# A4: FFA + aggregate_proj + transition residual (S2 baseline: 1.194)
#     Tests whether residual helps even without the concat_proj final head
run_experiment "S3A4_FFA_AGGPROJ_RESID_D${DEPTH}" \
    "Stage 3: learned_proj + FFA(soft) + aggregate_proj [+transition residual+norm]" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode aggregate_proj \
    --mst-final-topk -1 \
    --mst-routing-aux-weight 0.01 \
    --mst-diversity-weight 0.0

# ============================================================================
# Group C: FFA-based new features (pre-norm, no residual for FFA)
# ============================================================================

# C1: Fewer, wider subs (N=4, d=128 same D=512, 4x capacity per sub)
run_experiment "S3C1_FFA_N4_D128_D${DEPTH}" \
    "Stage 3: N=4 d=128 + FFA + concat_proj (wider subs)" \
    --mst-n-subs 4 --mst-sub-dim 128 \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# C2: Shared FFN up-proj (256 inner = same as 4*d, saves params)
run_experiment "S3C2_FFA_SHARED_FFN256_D${DEPTH}" \
    "Stage 3: FFA + concat_proj + shared FFN up (inner=256)" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-ffn-shared-up 1 --mst-ffn-inner-dim 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# C3: Shared FFN up-proj (512 inner = 2x wider, more feature detectors)
run_experiment "S3C3_FFA_SHARED_FFN512_D${DEPTH}" \
    "Stage 3: FFA + concat_proj + shared FFN up (inner=512)" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-ffn-shared-up 1 --mst-ffn-inner-dim 512 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# C4: FFA every-other-layer (halves transition FLOPs)
run_experiment "S3C4_FFA_EVERY2_D${DEPTH}" \
    "Stage 3: FFA every 2nd layer + concat_proj" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-transition-every 2 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# C5: FFA + sub dropout (forces robustness/specialization)
run_experiment "S3C5_FFA_SUBDROP_D${DEPTH}" \
    "Stage 3: FFA + concat_proj + sub_dropout=0.1" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-sub-dropout 0.1 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# ============================================================================
# Group D: Aggdist-based new features (with transition residual)
# ============================================================================

# D1: Fewer, wider subs with aggdist (N=4, d=128)
run_experiment "S3D1_AGGDIST_N4_D128_D${DEPTH}" \
    "Stage 3: N=4 d=128 + aggdist + concat_proj + residual" \
    --mst-n-subs 4 --mst-sub-dim 128 \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# D2: Shared FFN 512 + aggdist
run_experiment "S3D2_AGGDIST_SHARED_FFN512_D${DEPTH}" \
    "Stage 3: aggdist + concat_proj + residual + shared FFN 512" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-ffn-shared-up 1 --mst-ffn-inner-dim 512 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# D3: Global residual stream + aggdist
run_experiment "S3D3_AGGDIST_GLOBAL_RESID_D${DEPTH}" \
    "Stage 3: aggdist + concat_proj + residual + D-dim global stream" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-global-residual 1 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# D4: Sub dropout + aggdist
run_experiment "S3D4_AGGDIST_SUBDROP_D${DEPTH}" \
    "Stage 3: aggdist + concat_proj + residual + sub_dropout=0.1" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-sub-dropout 0.1 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# ============================================================================
# Group E: Temperature (FFA-specific)
# ============================================================================

# E1: FFA with sharper routing (temp=0.5)
run_experiment "S3E1_FFA_TEMP05_D${DEPTH}" \
    "Stage 3: FFA + concat_proj + temperature=0.5 (sharper routing)" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-ffa-temperature 0.5 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# ============================================================================
# Group F: Architectural (hybrid dense-MST, cross-sub KV sharing)
# ============================================================================

# F1: Hybrid dense-MST (even=dense D=512, odd=MST 8×64)
# Dense layers inject full-width capacity; MST layers specialize
run_experiment "S3F1_HYBRID_DENSE_D${DEPTH}" \
    "Stage 3: hybrid dense-MST + aggdist + concat_proj" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-hybrid-dense 1 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# F2: Cross-sub KV sharing (all subs share K,V projections, Q per-sub)
# Saves params, creates shared feature space for keys/values
run_experiment "S3F2_CROSS_SUB_KV_D${DEPTH}" \
    "Stage 3: FFA + concat_proj + cross-sub KV sharing" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-cross-sub-kv 1 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# ============================================================================
# Group G: FLOPs-controlled N=4 d=128 + FFN variants
# ============================================================================
# N=4 d=128 achieves 1.035-1.042 but FLOPs (3.43e8) exceed dense budget (2.86e8).
# Reducing FFN inner_dim from 4d=512 to 2d=256 cuts FFN FLOPs in half.

# G1: N=4 d=128 + reducedFFN(256) + FFA
run_experiment "S3G1_N4_RFFN256_FFA_D${DEPTH}" \
    "Stage 3: N=4 d=128 + FFA + reducedFFN(inner=256)" \
    --mst-n-subs 4 --mst-sub-dim 128 \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-ffn-inner-dim 256 \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# G2: N=4 d=128 + reducedFFN(256) + aggdist + residual
run_experiment "S3G2_N4_RFFN256_AGG_D${DEPTH}" \
    "Stage 3: N=4 d=128 + aggdist + reducedFFN(inner=256) + residual" \
    --mst-n-subs 4 --mst-sub-dim 128 \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode standard \
    --mst-ffn-inner-dim 256 \
    --mst-transition-mode aggregate_distribute \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# G3: N=4 d=128 + linear FFN (d→d, no expansion) + FFA
run_experiment "S3G3_N4_LINEAR_FFA_D${DEPTH}" \
    "Stage 3: N=4 d=128 + FFA + linear FFN (d→d)" \
    --mst-n-subs 4 --mst-sub-dim 128 \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode linear \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# G4: N=8 d=64 + linear FFN + FFA (minimal sub-transformer)
run_experiment "S3G4_N8_LINEAR_FFA_D${DEPTH}" \
    "Stage 3: N=8 d=64 + FFA + linear FFN (d→d)" \
    --mst-input-mode learned_proj \
    --mst-routing-mode soft_weighted --mst-routing-topk 4 --mst-ffn-mode linear \
    --mst-transition-mode free_for_all \
    --mst-final-mode concat_proj --mst-final-topk 0 \
    --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0

# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  P03 MST Stage 3 Sweep Complete — Depth ${DEPTH}"
echo "  Total experiments: 20"
echo "  A(4): transition residual  C(5): FFA features  D(4): aggdist features"
echo "  E(1): temperature  F(2): hybrid/crossKV  G(4): FLOPs-controlled N=4"
echo "═══════════════════════════════════════════════════════════════"
