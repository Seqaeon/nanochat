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
SHOW_STATUS=0
REDO=""
RUN_GROUPS="a b c d e"   # f and g are opt-in: run it explicitly at depth 12
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)  FORCE=1; shift ;;
        --status) SHOW_STATUS=1; shift ;;
        --redo)   REDO="$REDO $2"; shift 2 ;;
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

# The state file records a FINGERPRINT of the exact flags each tag ran with, not
# just the tag. A bare tag list cannot say whether an arm still means what it
# meant when it was marked done, so editing a sweep definition kept silently
# skipping the new version. Three guards now: the fingerprint must match, the run
# directory must exist, and --status / --redo let you inspect and clear entries.
done_already() {   # done_already <tag> <fingerprint>
    [[ "$FORCE" -eq 1 ]] && return 1
    if [[ ! -d "${OUT_BASE}/$1" ]]; then
        say "WARN  $1: state says done but ${OUT_BASE}/$1 has no artifacts. Re-running."
        return 1
    fi
    TAG="$1" FP="$2" python3 - <<'PYEOF'
import json, os, sys
try: s = json.load(open(os.environ['STATEFILE']))
except Exception: sys.exit(1)
e = s.get('completed', {})
e = dict.fromkeys(e, '') if isinstance(e, list) else e   # migrate the old list format
tag, fp = os.environ['TAG'], os.environ['FP']
if tag not in e: sys.exit(1)
sys.exit(0 if e[tag] in ('', fp) else 1)   # '' is a pre-fingerprint entry: trust it
PYEOF
}
mark_done() {   # mark_done <tag> <fingerprint>
    TAG="$1" FP="$2" python3 - <<'PYEOF'
import json, os
p = os.environ['STATEFILE']
try: s = json.load(open(p))
except Exception: s = {}
e = s.get('completed', {})
e = dict.fromkeys(e, '') if isinstance(e, list) else e
e[os.environ['TAG']] = os.environ['FP']
s['completed'] = e
json.dump(s, open(p + '.tmp', 'w'), indent=2); os.rename(p + '.tmp', p)
PYEOF
}
show_status() {
    python3 - <<'PYEOF'
import json, os
try: s = json.load(open(os.environ['STATEFILE']))
except Exception: print("  (no state file yet)"); raise SystemExit
e = s.get('completed', {})
e = dict.fromkeys(e, '') if isinstance(e, list) else e
if not e: print("  (nothing marked done)")
for k in sorted(e):
    ok = os.path.isdir(os.path.join(os.environ.get('OUT_BASE', ''), k))
    print(f"  {k:38s} {'artifacts present' if ok else 'NO ARTIFACTS -> will re-run'}"
          f"{'  (no fingerprint)' if not e[k] else ''}")
PYEOF
}
clear_tag() {
    TAG="$1" python3 - <<'PYEOF'
import json, os
p = os.environ['STATEFILE']
try: s = json.load(open(p))
except Exception: raise SystemExit
e = s.get('completed', {})
e = dict.fromkeys(e, '') if isinstance(e, list) else e
if e.pop(os.environ['TAG'], None) is not None: print("  cleared " + os.environ['TAG'])
s['completed'] = e
json.dump(s, open(p + '.tmp', 'w'), indent=2); os.rename(p + '.tmp', p)
PYEOF
}

# STATEFILE and OUT_BASE are read from the environment by the python helpers above.
export STATEFILE OUT_BASE

for _t in $REDO; do clear_tag "$_t"; done
if [[ "$SHOW_STATUS" == 1 ]]; then
    echo "state: $STATEFILE"
    show_status
    exit 0
fi

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

# Status lines go through the log too. Sending them to bare stdout while training
# output went through `tee` made them appear out of order relative to the run they
# describe, which reads as arms executing in the wrong sequence.
say() { echo "$@" | tee -a "$LOGFILE"; }

run() {   # run <tag> <common> <extra flags...>
    local tag="$1"; shift
    local common="$1"; shift
    local fp; fp=$(printf '%s|%s|%s|%s' "$common" "$HORIZON" "$*" "$DEPTH" | cksum | cut -d' ' -f1)
    if done_already "$tag" "$fp"; then say "SKIP  $tag (done, flags unchanged)"; return 0; fi
    say ""
    say "=============================================================="
    say "  $tag"
    say "  flags: $*"
    say "=============================================================="
    local dir="${OUT_BASE}/${tag}"
    [[ "$FORCE" == 1 && -d "$dir" ]] && rm -rf "$dir"
    if bash scripts/research_sweep.sh $common $HORIZON --out-dir "$dir" "$@" \
         $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        say "OK    $tag"; mark_done "$tag" "$fp"
    else
        say "FAIL  $tag, will retry next run"
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
#if [[ "$RUN_GROUPS" == *a* ]]; then
#    run "A0_dense_d${DEPTH}"      "$BASE_COMMON"
#    run "A1_remix_K8_d${DEPTH}"     "$REMIX_COMMON" --p22-n-templates 8
#    # run "A2_remix_K2_d${DEPTH}"     "$REMIX_COMMON" --p22-n-templates 2
#    run "A3_cond_R256_d${DEPTH}"    "$COND_COMMON"  --cond-rank 256
#fi

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
#if [[ "$RUN_GROUPS" == *d* ]]; then
#    run "D1_cond_tied_d${DEPTH}"    "$COND_COMMON" --cond-rank 256 --cond-gate-source tied
#    run "D2_cond_chunk256_d${DEPTH}" "$COND_COMMON" --cond-rank 256 --cond-chunk-size 256
#    run "D3_cond_linear_d${DEPTH}"  "$COND_COMMON" --cond-rank 256 --cond-coeff-act linear
#    run "D4_cond_ctx_d${DEPTH}"     "$COND_COMMON" --cond-rank 256 --cond-gate-source ctx
#fi

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
#if [[ "$RUN_GROUPS" == *e* ]]; then
#    run "E1_dense_swiglu_d${DEPTH}"    "$BASE_COMMON" --p36-swiglu-ffn 1
#    run "E2_cond_linear_branch_d${DEPTH}" "$COND_COMMON" --cond-rank 256 \
#        --cond-gate-source tied --cond-coeff-act one
#    run "E3_cond_tied_ffn_d${DEPTH}"   "$COND_COMMON" --cond-rank 256 \
#        --cond-gate-source tied --cond-sites ffn
#    run "E4_cond_tied_attn_d${DEPTH}"  "$COND_COMMON" --cond-rank 256 \
#        --cond-gate-source tied --cond-sites attn
#    run "E5_cond_tied_sigmoid_d${DEPTH}" "$COND_COMMON" --cond-rank 256 \
#        --cond-gate-source tied --cond-coeff-act sigmoid
#fi
#
# ── F. the >200M regime test ────────────────────────────────────────────────
#   bash scripts/p35_conditioned_sweep.sh 12 --group f
#
# Every variant measured so far is sub-200M (d4 = 36.7M, d8 = 125.8M); only dense
# has a >200M point. Efficiency also falls with R (R/D 0.25 -> 1.09x, 0.50 ->
# 1.04x, 1.00 -> 0.92x, 2.00 -> 0.74x, floor-corrected) and attention-only beats
# both-sites per unit spent, so R=192 at d12 puts R/D back at 0.25, the
# best-tested ratio, combined with the best-performing site restriction. That
# pairing has never been run at any depth.
#
# F1 costs about 1.21x the dense d12 budget.
# SUCCESS THRESHOLD: BPB below ~0.857. Dense at that compute is 0.8577 by the
# floor-corrected 3-point fit and 0.8567 by the local d8->d12 slope of -0.0542.
#
# Run F2 ONLY if F1 clears the threshold. It is the exact matched control:
# identical params, tokens and FLOPs, so the nonlinearity comparison needs no
# scaling assumption. That mechanism was worth 0.0285 BPB at d4 and 0.0142 at d8.
#if [[ "$RUN_GROUPS" == *f* ]]; then
#    run "F1_cond_attn_R192_d${DEPTH}" "$COND_COMMON" --cond-rank 192 \
#        --cond-gate-source tied --cond-sites attn
#    run "F2_cond_attn_R192_linear_d${DEPTH}" "$COND_COMMON" --cond-rank 192 \
#        --cond-gate-source tied --cond-sites attn --cond-coeff-act one
#fi

# ── G. FFN allocation across depth ──────────────────────────────────────────
#   bash scripts/p35_conditioned_sweep.sh 8 --group g
#
# All DENSE. This is a claim about where a dense transformer puts its own
# capacity, so it needs no conditioning machinery to test.
#
# scripts/conditioning_headroom.py on the dense d22 checkpoint measured, per
# layer, the demand (per-token gradient dispersion), the fraction of it a linear
# router could reach, and the supply (Jacobian dispersion the FFN nonlinearity
# already provides). Supply totals 0.81x reachable demand, so the model has
# roughly the right AMOUNT of FFN. It is in the wrong PLACES, by up to 29x:
#
#   layers 0-7, 10-12, 19-20   over-provisioned, up to 3.4x
#   layers 13-15, 17-18, 21    under-provisioned, up to 5.9x  (layer 18: 60 vs 355)
#
# All 22 layers had identical width, so that spread is a property of the
# activations, not of the architecture, and it is not circular.
#
#   G1  reallocate at FIXED parameters, proportional to measured reachable
#       demand. FLOP-neutral, so any BPB gain is free efficiency. Run this first:
#       it is the cleanest test of whether the measurement predicts anything.
#   G2  shrink only the over-provisioned layers. Removes compute, which is the
#       direction that actually moves the efficiency metric.
#   G3-G7  how much FFN is needed at all, and whether placement or total matters.
#       The plain variants cut parameters; the _iso variants hold them fixed and
#       concentrate the same width into fewer blocks.
#if [[ "$RUN_GROUPS" == *g* ]]; then
#    run "G1_ffn_measured_d${DEPTH}"    "$BASE_COMMON" --p34-ffn-schedule measured
#    run "G2_ffn_shrink_d${DEPTH}"      "$BASE_COMMON" --p34-ffn-schedule shrink
#    run "G3_ffn_last_only_d${DEPTH}"   "$BASE_COMMON" --p34-ffn-schedule last
#    run "G4_ffn_every2_d${DEPTH}"      "$BASE_COMMON" --p34-ffn-schedule every2
#    run "G5_ffn_every2_iso_d${DEPTH}"  "$BASE_COMMON" --p34-ffn-schedule every2_iso
#    run "G6_ffn_every4_d${DEPTH}"      "$BASE_COMMON" --p34-ffn-schedule every4
#    run "G7_ffn_every4_iso_d${DEPTH}"  "$BASE_COMMON" --p34-ffn-schedule every4_iso
#    # G3 extensions: rescue the last-only skeleton with cheap replacements
#    #   G3a: D->D linear in FFN-less layers (tests recombination vs nonlinearity)
#    run "G3a_ffn_last_linear_d${DEPTH}" "$BASE_COMMON" --p34-ffn-schedule last --p34-ffn-no-ffn-replacement linear
#    #   G3b: deep final FFN D->4D->4D->D (tests depth-in-one-spot)
#    run "G3b_ffn_last_deep_d${DEPTH}"   "$BASE_COMMON" --p34-ffn-schedule last --p34-ffn-last-depth 2
#    #   G3c: bottleneck final FFN D->D->4D->D (cheapest depth extension)
#    run "G3c_ffn_last_neck_d${DEPTH}"   "$BASE_COMMON" --p34-ffn-schedule last --p34-ffn-last-depth 3
#fi

#
# ── H. headroom-informed experiments ─────────────────────────────────────────
#   bash scripts/p35_conditioned_sweep.sh 8 --group h
#
# Guided by conditioning_headroom.py run on actual d8/d12/d20/d22 checkpoints
# (headroom_results.log). Key findings that inform these experiments:
#
#   1. The d22-interpolated profile used by G1 was ANTI-CORRELATED with the
#      actual d8 demand. Native d8 profile is: early+layer6 = high demand,
#      layer3 = lowest.  d22 said the opposite.
#   2. attn.c_v has DOF ~7-12 at EVERY depth (top1 ~0.47), nearly rank-1.
#      E4 wastes 25% of its R budget on a projection whose ceiling is ~10 DOF.
#   3. FFN excess alignment (supply meeting demand) is lowest in early layers
#      and highest in late layers, consistently across all depths.
#      => attention nonlinearity helps MORE in early layers.
#   4. c_proj (DOF 82-265) and c_q (DOF 132-176) have the highest attention
#      demand.  c_k (DOF 67-89) is moderate.
#
#   H1  G1 with the CORRECT profile: measured on the target depth's own ckpt.
#   H2  E4 skipping c_v entirely.  Same R, 25% fewer conditioned projections.
#   H3  E4 early-layers-only: conditioning only the first half of layers where
#       FFN supply is poorly aligned and unmet demand is highest.
#   H4  E4 on c_proj + c_q only: concentrate R on the two highest-DOF projections.
#   H5  Orthogonal combination: E4 (targeted) on attention + G1 (native) on FFN.
if [[ "$RUN_GROUPS" == *h* ]]; then
    # H1: FFN reallocation with depth-native profile
    run "H1_ffn_native_d${DEPTH}"       "$BASE_COMMON" --p34-ffn-schedule measured_native
    # H2: E4 without c_v (skip the projection with DOF ~7)
    run "H2_cond_skip_v_d${DEPTH}"      "$COND_COMMON" --cond-rank 256 \
        --cond-gate-source tied --cond-sites attn --cond-attn-projs qko
    # H3: E4 early-layers-only (first half has lowest FFN excess)
    run "H3_cond_early_half_d${DEPTH}"  "$COND_COMMON" --cond-rank 256 \
        --cond-gate-source tied --cond-sites attn --cond-layer-frac 0.5
    # H4: E4 on c_proj + c_q only (highest DOF projections)
    run "H4_cond_qo_only_d${DEPTH}"    "$COND_COMMON" --cond-rank 256 \
        --cond-gate-source tied --cond-sites attn --cond-attn-projs qo
    # H5: orthogonal stack: targeted E4 + native G1 FFN reallocation
    run "H5_cond_qko_native_ffn_d${DEPTH}" "$COND_COMMON" --cond-rank 256 \
        --cond-gate-source tied --cond-sites attn --cond-attn-projs qko \
        --p34-ffn-schedule measured_native
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

