#!/bin/bash
#
# P33: the ablations the reviewers and the AC asked for, at d4.
#
# The abstract names three design choices as what "enable dynamic linear layers
# to outperform static baselines": chunk-amortized routing, quantile balancing,
# and identity-preserving init. Table 4 ablates only the first. Quantile
# balancing has no ablation row anywhere in the paper, and the gate is ablated
# only at K=1 where it contributes 0.002-0.005 BPP. Two of three headline
# contributions currently rest on the phase narrative rather than a controlled
# experiment at the operating point. These groups fix that.
#
#   A  quantile balancing at K=8      3 conditions   (no row exists in the paper)
#   B  output gate at K=8             3 conditions   (paper ablates at K=1 only)
#   C  seed variance                  6 runs         (4 reviewers + the AC flagged n=1)
#   D  LayerNorm confound             2 conditions   (3 reviewers + the AC)
#
# Group A is ordered first on purpose. Before running anything, know that
# QuantileBalancedRouter is provably a no-op: it unions the quantile-thresholded
# set with the per-position top-k, and since the top-k entries hold the largest
# scores the subsequent re-selection returns exactly those same entries. Measured
# bit-identical to plain top-k plus softmax across topk in {0,1,2,4} x {train,
# eval}. So A_quantile and A_plain are expected to produce the *same numbers*.
# If they do, that is the result: the mechanism the paper credits does nothing,
# and the honest ablation row says so. If they differ, something in the training
# loop is doing more than the router and it needs finding.
#
# Also worth confirming before you interpret anything: REMIX_COMMON in
# p29_sweep.sh sets --p23-quantile-route 1, but the live 29C arm overrides it to
# 0. Check which value the published d4/d8/d12 runs actually used, because the
# paper's Section 3 describes quantile routing with an EMA threshold (lambda=0.99)
# and the implementation has no EMA at all.
#
#   bash scripts/p33_rebuttal_ablations.sh                 # everything at d4
#   bash scripts/p33_rebuttal_ablations.sh --group a       # one group
#   bash scripts/p33_rebuttal_ablations.sh --seeds 3 8     # depth override
#   bash scripts/p33_rebuttal_ablations.sh --force         # ignore saved state
#
set -o pipefail

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-out/.triton_cache}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"

FORCE=0
RUN_GROUPS="a b c d"
SEEDS="0 1 2"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)  FORCE=1; shift ;;
        --group)  RUN_GROUPS="$2"; shift 2 ;;
        --seed-list) SEEDS="$2"; shift 2 ;;
        [0-9]*)   DEPTH="$1"; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done
DEPTH="${DEPTH:-4}"

ASPECT_RATIO="${ASPECT_RATIO:-64}"
MODEL_DIM=$(python3 -c "d=$DEPTH; ar=$ASPECT_RATIO; h=128; print(((d*ar+h-1)//h)*h)")
OUT_BASE="${OUT_BASE:-out/sweep_p33}"
LOGFILE="${SWEEP_LOG:-sweep_p33_d${DEPTH}.log}"
STATEFILE="${LOGFILE%.log}_state.json"
[[ "$FORCE" == 1 ]] && rm -f "$STATEFILE"
[[ -f "$STATEFILE" ]] || echo '{"completed":[]}' > "$STATEFILE"

done_already() {
    [[ "$FORCE" -eq 1 ]] && return 1
    python3 -c "
import json,sys
try: s=json.load(open('$STATEFILE'))
except Exception: sys.exit(1)
sys.exit(0 if '$1' in s.get('completed',[]) else 1)" 2>/dev/null
}
mark_done() {
    python3 - <<PYEOF
import json, os
try: s=json.load(open('$STATEFILE'))
except Exception: s={'completed':[]}
if '$1' not in s.setdefault('completed',[]): s['completed'].append('$1')
json.dump(s, open('${STATEFILE}.tmp','w'), indent=2); os.rename('${STATEFILE}.tmp','$STATEFILE')
PYEOF
}

# Identical to REMIX_COMMON in p29_sweep.sh except that --p23-quantile-route is
# NOT set here: every group below sets it explicitly, because leaving it implicit
# is how the paper ended up describing a router the runs may not have used.
REMIX_COMMON="--fp8 --max-shards ${MAX_SHARDS:-170} --models remixed-linear \
  --device-batch-size ${REMIX_DEVICE_BATCH_SIZE:-${DEVICE_BATCH_SIZE:-16}} \
  --total-batch-size -1 --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --aspect-ratio $ASPECT_RATIO \
  --target-param-data-ratio 10.5 --warmup-ratio 0.005 --warmdown-ratio 0.65 \
  --final-lr-frac 0.05 --research-dim -1 --remix-basis-size $MODEL_DIM \
  --cclblock-modulation weight --cclblock-context-stream selective \
  --cclblock-gate-temperature 2.0 --remix-shared-context-gates 0 \
  --remix-use-context 1 --p22-template-routing-learned 1 \
  --remix-use-basis-gate 0 --remix-basis-gate-mode centered \
  --target-tokens -1 --target-active-params 0 --save-every 200 \
  --p28-chunk-routing-size ${CHUNK:-256} --p22-n-templates 8 --p22-template-topk 0"

BASE_COMMON="--fp8 --max-shards ${MAX_SHARDS:-170} --models base \
  --device-batch-size ${BASE_DEVICE_BATCH_SIZE:-${DEVICE_BATCH_SIZE:-32}} \
  --total-batch-size -1 --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --aspect-ratio $ASPECT_RATIO \
  --target-param-data-ratio 10.5 --warmup-ratio 0.005 --warmdown-ratio 0.65 \
  --final-lr-frac 0.05 --research-dim -1 --target-tokens -1 \
  --target-active-params 0 --save-every 200"

run() {   # run <tag> <common> <extra flags...>
    local tag="$1"; shift
    local common="$1"; shift
    if done_already "$tag"; then echo "⏭  $tag (done)"; return 0; fi
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  $tag"
    echo "╚══════════════════════════════════════════════════════════════╝"
    local dir="${OUT_BASE}/${tag}"
    [[ "$FORCE" == 1 && -d "$dir" ]] && rm -rf "$dir"
    if bash scripts/research_sweep.sh $common --out-dir "$dir" "$@" \
         $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $tag"; mark_done "$tag"
    else
        echo "❌  $tag FAILED — will retry next run"
    fi
}

echo "════════════════════════════════════════════════════════════"
echo "  P33 rebuttal ablations   depth ${DEPTH}  D=${MODEL_DIM}"
echo "  groups: ${RUN_GROUPS}   seeds: ${SEEDS}   chunk: ${CHUNK:-256}"
echo "════════════════════════════════════════════════════════════"

# ── A. Quantile balancing at K=8 ────────────────────────────────────────────
# The claim under test: "eliminating the auxiliary losses required by standard
# MoE". Report template-utilisation entropy alongside BPB, via
# scripts/paper_template_analysis.py on each checkpoint. A_plain vs A_quantile is
# the row the paper needs and does not have.
if [[ "$RUN_GROUPS" == *a* ]]; then
    run "A1_route_plain_K8_d${DEPTH}"     "$REMIX_COMMON" --p23-quantile-route 0
    run "A2_route_quantile_K8_d${DEPTH}"  "$REMIX_COMMON" --p23-quantile-route 1
    run "A3_route_xattn_K8_d${DEPTH}"     "$REMIX_COMMON" --p23-quantile-route 2
fi

# ── B. Output gate at K=8 ───────────────────────────────────────────────────
# The paper ablates the gate only at K=1, where it is worth 0.002-0.005 BPP, then
# calls it one of three key design choices. This measures it at the shipped
# configuration. 'centered' is the 1+tanh identity-preserving gate; 'linear' is
# the same shape without the centering; B3 removes the gate entirely.
if [[ "$RUN_GROUPS" == *b* ]]; then
    run "B1_gate_centered_K8_d${DEPTH}" "$REMIX_COMMON" --p23-quantile-route 0 \
        --remix-use-output-gate 1 --remix-basis-gate-mode centered
    run "B2_gate_linear_K8_d${DEPTH}"   "$REMIX_COMMON" --p23-quantile-route 0 \
        --remix-use-output-gate 1 --remix-basis-gate-mode linear
    run "B3_gate_none_K8_d${DEPTH}"     "$REMIX_COMMON" --p23-quantile-route 0 \
        --remix-use-output-gate 0 --remix-basis-gate-mode none
fi

# ── C. Seed variance ────────────────────────────────────────────────────────
# Gives you "seed sigma is X BPB, the d12 gap is Y sigma", which is a real
# argument. Right now there is none and four reviewers plus the AC noticed.
# NOTE --seed controls weight init only; the dataloader order is not seeded, so
# this is a lower bound on total run-to-run variance. Say that in the paper.
if [[ "$RUN_GROUPS" == *c* ]]; then
    for s in $SEEDS; do
        run "C_remix_seed${s}_d${DEPTH}" "$REMIX_COMMON" --p23-quantile-route 0 --seed "$s"
        run "C_dense_seed${s}_d${DEPTH}" "$BASE_COMMON"  --seed "$s"
    done
fi

# ── D. LayerNorm confound ───────────────────────────────────────────────────
# R3, R4, R5 and the AC all flag that RemixedLinear adds an intermediate
# LayerNorm the dense baseline does not have, so every headline gap includes it.
# D1 gives dense the same LayerNorm; D2 removes it from Remix. This can only
# clarify or hurt, and not running it is worse than either outcome: if D1 lands
# near 1.168 the factorization contributes nothing and routing is the whole
# story, which is a cleaner paper.
if [[ "$RUN_GROUPS" == *d* ]]; then
    run "D1_dense_plus_ln_d${DEPTH}"  "$BASE_COMMON"  --dense-intermediate-ln 1
    run "D2_remix_no_ln_K8_d${DEPTH}" "$REMIX_COMMON" --p23-quantile-route 0 \
        --remix-disable-ln-basis 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  P33 complete — results in ${OUT_BASE}, log ${LOGFILE}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Next: run scripts/paper_template_analysis.py on the A-group checkpoints to"
echo "get the utilisation entropy that turns A into a load-balancing ablation"
echo "rather than just a BPB row."
