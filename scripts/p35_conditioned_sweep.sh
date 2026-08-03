#!/bin/bash
#
# P35: ConditionedLinear, conditioning in activation space instead of weight space.
#
# Why this exists, in the order the evidence arrived:
#
#   1. The K=8 bank runs at ~25 FLOP/byte against an H200 ridge of 206, so it is
#      eight times inside the bandwidth-bound regime no matter how deep the model.
#   2. It stores K× parameters, and under a fixed data:param ratio parameters set
#      the token budget, so every K=8 vs dense comparison in the paper gave the
#      two arms different amounts of data.
#   3. It buys at most K-1 degrees of freedom per token: one coefficient per
#      template, each broadcast across a whole (out, basis) matrix. The low-rank
#      delta variant does not escape this: `weights_all` is (B, n_chunks, K) and
#      is broadcast across the rank dimension too.
#   4. scripts/paper_template_analysis.py on the trained d12 checkpoint says the
#      realised number is far below even that ceiling: 59/72 modules route to a
#      single template despite topk=0, layers 1-6 have |dW|/|W| = 0.000 and a
#      usage CV of 2.6458 = sqrt(K-1) exactly, i.e. total collapse. Mean K_eff
#      over projections is 1.52. Bank cosine stays low (0.02-0.23), so the
#      templates are distinct and simply never selected.
#
# So the bank is not underpowered, it is unused, and the mechanism that leaves it
# unused is the simplex: a softmax over K competing templates has a winner-take-
# all fixed point, and once one logit dominates the others receive vanishing
# gradient. ConditionedLinear removes the competition: R independent
# unnormalised coefficients, c(x) = 1 + tanh(W_r x), no softmax and no
# normalisation across coefficients, so every coefficient keeps its own gradient
# for the whole run.
#
#   additive        y = W0 x + U (c(x) * V^T x)                  c(x) in R^R
#   multiplicative  z_i = z_{i-1} + c_i(x) u_i (v_i^T z_{i-1}),  y = W0 z_m
#
# HORIZON POLICY: every arm gets its own Chinchilla-optimal budget, sized by
# ACTIVE parameters (--target-active-params 1), not by stored ones. That is the
# Pareto-correct rule and it is not the same as the stored-parameter rule the
# earlier sweeps used:
#
# Measured at d4 (D=256, vocab 65536, ratio 10.5):
#
#   arm                   stored    active    budget
#   dense                 70.2M     70.2M     209M tokens
#   remix K=8 chunk256    88.7M     69.8M     204M      bank amortizes over the
#   remix K=2 chunk256    74.5M     69.8M     204M      chunk, so ~255/256 of it
#   cond R=256            77.3M     77.3M     284M      counts inactive
#   cond m=16             70.7M     70.7M     214M
#
# The bank's 8x storage buys it no extra data at all under this rule, because
# active_per_token = template_params/chunk. The cond arms do get more, because
# they genuinely read every parameter for every token. That is the Pareto-correct
# answer, and it means BPB is comparable WITHIN an arm's own optimum but not
# across arms at face value. To compare across arms, use the 'Total active
# training FLOPs estimate' line base_train prints.
#
# The cond ratio grows with depth: at d4 the 16.8M lm_head dilutes it, and by d12
# the transformer matrices dominate and cond R=256 approaches ~1.7x dense.
#
# Read cond_dof_pr in <out-dir>/gate_stats.jsonl alongside BPB. It is the
# participation ratio of the coefficient covariance, the same measurement as
# K_eff generalised, and it is the number that says whether the conditioning
# survived training or collapsed the way the bank did.
#
#   bash scripts/p35_conditioned_sweep.sh                 # everything at d4
#   bash scripts/p35_conditioned_sweep.sh --group a c     # selected groups
#   bash scripts/p35_conditioned_sweep.sh 8               # depth override
#   bash scripts/p35_conditioned_sweep.sh --force         # ignore saved state
#
set -o pipefail

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-out/.triton_cache}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"

FORCE=0
RUN_GROUPS="a b c d e"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)  FORCE=1; shift ;;
        --group)  RUN_GROUPS="$2"; shift 2 ;;
        [0-9]*)   DEPTH="$1"; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done
DEPTH="${DEPTH:-4}"

ASPECT_RATIO="${ASPECT_RATIO:-64}"
MODEL_DIM=$(python3 -c "d=$DEPTH; ar=$ASPECT_RATIO; h=128; print(((d*ar+h-1)//h)*h)")
OUT_BASE="${OUT_BASE:-out/sweep_p35}"
LOGFILE="${SWEEP_LOG:-sweep_p35_d${DEPTH}.log}"
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

# Everything except the horizon is shared with p33's REMIX_COMMON, so a p35 arm
# and a p33 arm differ only in the layer under test.
SHARED="--fp8 --max-shards ${MAX_SHARDS:-170} \
  --total-batch-size -1 --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --aspect-ratio $ASPECT_RATIO \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 \
  --final-lr-frac 0.05 --research-dim -1 --target-active-params 1 --save-every 200"

BASE_COMMON="$SHARED --models base --device-batch-size ${BASE_DEVICE_BATCH_SIZE:-${DEVICE_BATCH_SIZE:-32}}"

REMIX_COMMON="$SHARED --models remixed-linear \
  --device-batch-size ${REMIX_DEVICE_BATCH_SIZE:-${DEVICE_BATCH_SIZE:-16}} \
  --remix-basis-size $MODEL_DIM \
  --cclblock-modulation weight --cclblock-context-stream selective \
  --cclblock-gate-temperature 2.0 --remix-shared-context-gates 0 \
  --remix-use-context 1 --p22-template-routing-learned 1 \
  --remix-use-basis-gate 0 --remix-basis-gate-mode centered \
  --p23-quantile-route 0 --p28-chunk-routing-size ${CHUNK:-256} --p22-template-topk 0"

# ConditionedLinear needs none of the bank machinery: no basis projection, no
# context stream, no template bank, no gate. Only the modulation mode and the
# cond flags.
COND_COMMON="$SHARED --models remixed-linear \
  --device-batch-size ${REMIX_DEVICE_BATCH_SIZE:-${DEVICE_BATCH_SIZE:-16}} \
  --remix-basis-size $MODEL_DIM --cclblock-modulation cond"

# Horizon: each arm sizes its own budget from its active parameter count. No
# arm is pinned to another's token count, so no arm is starved or overfed
# relative to its own compute-optimal point. TOKEN_BUDGET=<n> overrides for
# every arm if you do want a hard pin for a specific comparison.
HORIZON="--target-param-data-ratio ${RATIO:-10.5}"
[[ -n "$TOKEN_BUDGET" ]] && HORIZON="$HORIZON --target-tokens $TOKEN_BUDGET"

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
    if bash scripts/research_sweep.sh $common $HORIZON --out-dir "$dir" "$@" \
         $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $tag"; mark_done "$tag"
    else
        echo "❌  $tag FAILED, will retry next run"
    fi
}

echo "════════════════════════════════════════════════════════════"
echo "  P35 ConditionedLinear   depth ${DEPTH}  D=${MODEL_DIM}"
echo "  groups: ${RUN_GROUPS}   horizon: ${TOKEN_BUDGET:-per-arm, sized by active params}"
echo "════════════════════════════════════════════════════════════"

# ── A. the comparison the paper needs and does not have ─────────────────────
# A0 is commented out: the dense baseline at this depth is already measured, and
# under the per-arm horizon rule nothing downstream depends on re-running it.
# Re-enable it only if the depth, aspect ratio, ratio or data shards change,
# since all four of those move the dense number.
# A1 is the shipped design. A2 is the control the template analysis demands: at
# a measured K_eff of 1.52 the bank is behaving as K~2 already, so if A2 matches
# A1 the extra six templates are pure storage. A3 is the new layer at the R the
# DOF table points to.
if [[ "$RUN_GROUPS" == *a* ]]; then
    run "A0_dense_d${DEPTH}"      "$BASE_COMMON"
#    run "A1_remix_K8_d${DEPTH}"     "$REMIX_COMMON" --p22-n-templates 8
    # run "A2_remix_K2_d${DEPTH}"     "$REMIX_COMMON" --p22-n-templates 2
#    run "A3_cond_R256_d${DEPTH}"    "$COND_COMMON"  --cond-rank 256
fi

# ── B. how much conditioning is actually useful ─────────────────────────────
# R is a free capacity knob now: it is decoupled from storage layout and costs
# R(d_in+d_out) parameters instead of R·d_in·d_out. If BPB keeps improving to
# R=512 the bank was capacity-starved all along; if it saturates at R=64 the
# answer is that conditioning was never worth many bits and the whole family
# should be cheap.
#if [[ "$RUN_GROUPS" == *b* ]]; then
#    run "B1_cond_R64_d${DEPTH}"     "$COND_COMMON" --cond-rank 64
#    run "B2_cond_R128_d${DEPTH}"    "$COND_COMMON" --cond-rank 128
#    run "B3_cond_R512_d${DEPTH}"    "$COND_COMMON" --cond-rank 512
#    # Router is ~1/3 of the added parameters at R=256. If the factored router
#    # matches, spend the difference on R instead.
#    run "B4_cond_R256_rr64_d${DEPTH}" "$COND_COMMON" --cond-rank 256 --cond-router-rank 64
#fi

# ── C. multiplicative composition: m against R at matched parameters ────────
# Additive gives the span of R rank-1 terms; sequential composition gives the
# group they generate. m=16 costs 2·m·d_in parameters, 1.06x dense, so C1-C3
# are the cheapest arms in the sweep by a wide margin. C4 is the honest test of
# whether composition adds anything the same parameter count spent additively
# does not: m=16 costs about what R=12 costs, so C4 vs B1 is the comparison.

# The C arms produced NaN in the first p35 run. The composition was unbounded:
# nothing constrained ||u||, and the recursion amplifies geometrically in m. It
# is now a proper exponential map (u and v unit-normalized, |c| <= 1/m), so every
# factor has spectral norm <= 1 + 1/m and the product is bounded by e for any m.
# Identity at init moved from u=0 to c=0, which also keeps the router's gradient
# live at step 0. Re-run these with --force.
#if [[ "$RUN_GROUPS" == *c* ]]; then
#    run "C1_cond_m8_d${DEPTH}"      "$COND_COMMON" --cond-rank 0 --cond-mult-steps 8
#    run "C2_cond_m16_d${DEPTH}"     "$COND_COMMON" --cond-rank 0 --cond-mult-steps 16
#    run "C3_cond_m32_d${DEPTH}"     "$COND_COMMON" --cond-rank 0 --cond-mult-steps 32
#    run "C4_cond_R64_m16_d${DEPTH}" "$COND_COMMON" --cond-rank 64 --cond-mult-steps 16
#fi

# ── D. what the gain is actually made of ────────────────────────────────────
# D1 is the one that matters most. 'tied' reuses V^T x as its own gate, so there
# is no independent conditioning signal at all and the layer is a plain gated
# low-rank branch, a GLU relative. If D1 matches A3, the win is nonlinearity
# and not conditioning, and that is the honest finding.
# D2 answers whether per-token routing was ever needed: the template analysis
# found 29/72 modules with zero within-sequence routing variance, so chunk size
# was already inert in 40% of the bank's modules.
# D3 removes the bound on c, letting a coefficient flip sign.
# D4 conditions on the attention output rather than the layer's own input.
if [[ "$RUN_GROUPS" == *d* ]]; then
    run "D1_cond_tied_d${DEPTH}"    "$COND_COMMON" --cond-rank 256 --cond-gate-source tied
#    run "D2_cond_chunk256_d${DEPTH}" "$COND_COMMON" --cond-rank 256 --cond-chunk-size 256
#    run "D3_cond_linear_d${DEPTH}"  "$COND_COMMON" --cond-rank 256 --cond-coeff-act linear
#    run "D4_cond_ctx_d${DEPTH}"     "$COND_COMMON" --cond-rank 256 --cond-gate-source ctx
fi

# ── E. what the p35 winner actually is ──────────────────────────────────────
# D1 (cond tied) won at 1.53x dense FLOP efficiency. But c = 1 + tanh(t) applied
# to t = V^T x means the branch is exactly t*(1 + tanh(t)) = SiLU(2t), so D1 is
#
#     y = W0 x + U * SiLU(2 V^T x)
#
# a parallel low-rank SiLU MLP branch on every projection. Not a conditioned
# operator, and not even a GLU (a GLU needs two projections, which is the
# full-router arm that scored WORSE at 1.42x). This group decides what to call it.
#
#   E1  the ordinary way to add a gated nonlinearity to a transformer. If this
#       matches D1 then the FFN activation was the whole story and there is no
#       architecture result.
#   E2  strips the nonlinearity: c = 1 exactly, so W_eff = W0 + U V^T merges into
#       a single dense matrix and the model is dense in every mathematical sense.
#       If E2 matches D1 the effect is optimization, not expressivity, and the
#       finding is about over-parameterized reparameterization.
#   E3/E4  localize it. The FFN already has a nonlinearity and attention's
#       Q/K/V/O do not, so this separates "the FFN wanted a better activation"
#       from "the projections wanted one at all".
#   E5  D1 with ReLU^2 instead of SiLU: is the specific activation load-bearing?
if [[ "$RUN_GROUPS" == *e* ]]; then
    run "E1_dense_swiglu_d${DEPTH}"    "$BASE_COMMON" --p36-swiglu-ffn 1
    run "E2_cond_linear_branch_d${DEPTH}" "$COND_COMMON" --cond-rank 256 \
        --cond-gate-source tied --cond-coeff-act one
#    run "E3_cond_tied_ffn_d${DEPTH}"   "$COND_COMMON" --cond-rank 256 \
#        --cond-gate-source tied --cond-sites ffn
    run "E4_cond_tied_attn_d${DEPTH}"  "$COND_COMMON" --cond-rank 256 \
        --cond-gate-source tied --cond-sites attn
#    run "E5_cond_tied_sigmoid_d${DEPTH}" "$COND_COMMON" --cond-rank 256 \
#        --cond-gate-source tied --cond-coeff-act sigmoid
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  done, depth ${DEPTH}"
echo ""
echo "  Collect BPB:"
echo "    python -m scripts.paper_collect --run-dir ${OUT_BASE}"
echo "  Conditioning actually used, per arm (the K_eff analogue):"
echo "    python -c \"import json;rows=[json.loads(l) for l in open('${OUT_BASE}/<tag>/gate_stats.jsonl')];\\"
echo "      print(rows[-1]['cond_dof_pr'], rows[-1]['cond_tok_std'])\""
echo "  A cond_dof_pr near R means the coefficients stayed independent; near 1"
echo "  means it collapsed the same way the K=8 router did, and the design does"
echo "  not work either. That is the result to report either way."
echo "════════════════════════════════════════════════════════════"
