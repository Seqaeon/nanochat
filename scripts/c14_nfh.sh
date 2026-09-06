#!/usr/bin/env bash
# ============================================================================
# C14: nonnegative factorized heads. Reparameterise the DISTRIBUTION, not the matrix.
#
# WHY EVERYTHING BEFORE THIS TOPPED OUT
#   c00-c13 all approximate the V x d head MATRIX and normalise exactly. Three
#   measurements, taken on the trained d8 V=131,072 head, say that route is closed:
#
#   1. A head linear in h has logit rank <= d, which dense already attains, so a
#      cheaper linear head is a constrained rank-d matrix and the constraint IS the
#      quality cost.
#   2. The head's rows do not cluster. In the whitened metric, where Euclidean
#      distance is RMS logit error over the real activations, k-means radius falls
#      only 2.455 -> 1.983 from k=16 to k=4096 and tracks a Gaussian control of the
#      same second moment at a flat 0.89 ratio. Six levels of RESIDUAL quantisation
#      (256^6 cells against V=131k) move it from 3.73 only to 1.74, also tracking
#      the control. That kills product codes, VQ-Logits and RQ-VAE semantic IDs at
#      every quantisation depth, and it also kills the class score of a two-level
#      head: fitting log sum_{w in c} exp(W_w h) by a linear map leaves test
#      R^2 = 0.927 and costs 1.977 nats = 0.568 bpb at 512 clusters.
#   3. Materialising V logits is a bandwidth floor. Dense and a perfectly fused
#      cheap head write the same (N, V) tensor, so the ceiling is ~2.6x however much
#      arithmetic is removed (OPEN_QUESTIONS Q8).
#
# WHAT THIS DOES INSTEAD
#   p(w|h) = sum_r pi_r(h) prod_g alpha^g_{r, a_g(w)}(h),  every factor a softmax.
#
#   The partition function is 1 by construction, no (N, V) tensor exists on the
#   training path at all, the whole head is two or three dense GEMMs, and because
#   log p is a log-sum-exp of sums of log-softmaxes the log-prob matrix is NOT
#   capped at rank d+1. Cheaper than dense AND above its rank ceiling, which is the
#   pairing Mixture of Softmaxes buys at R x dense cost.
#
#   THIS IS THE BINARY CODE HEAD, FIXED. a_g is a K-ary digit and alpha^g its score,
#   so R=1 is exactly the independent-digit head: LightRNN at G=2, Oda et al. at
#   K=2. Phase 0 answered R=1's rank-1 limitation by raising the code's INTERACTION
#   ORDER, which costs sum_{j<=k} C(G,j)(K-1)^j columns. A mixture reaches every
#   distribution of nonnegative tensor rank <= R for R*sum_g K_g columns. Coupling is
#   exponentially cheaper in probability space than in logit space, and that is why
#   the order-3 to order-4 rung cost 2.7x the FLOPs and bought nothing.
#
# OFFLINE ORACLE, measured on the d8 V=131,072 head (acts_d8_v131k.pt) with a nested
# assignment, against the +0.0931 bpb that a head costing NOTHING AT ALL is worth at
# depth 8. These are FREE fits, so for cp they are a lower bound on the excess:
#
#     mode            R        excess bpb    head MACs vs dense
#     cp 64x64x32     1          0.2829         0.0012x     <- LightRNN
#     cp 64x64x32     8          0.0325         0.0098x
#     cp 64x64x32     16         0.0170         0.0197x
#     cp 64x64x32     32         0.0097         0.0393x     <- headline
#     cp 64x64x32     64         0.0056         0.0786x
#     cp 512x256      32         0.0033         0.188x
#     global          m=256      0.0573         0.00196x
#
#   Those are FREE fits, so for cp they are a lower bound on the excess: the real
#   head must produce 5,152 factor parameters per token through a 512-dimensional h.
#   That gap is the one thing this run exists to measure. NFH_global is the hedge
#   precisely because its per-context degrees of freedom (256) sit BELOW d.
#
# WHAT THE FIRST d8 SWEEP SAID, and what it changed
#   R=1 1.5101, R=32 1.1220, dense 0.9691. The 0.388 bpb between R=1 and R=32 is the
#   claim and it survived: the mixture IS the mechanism. But R=32 came in +0.1529 bpb
#   above dense against a +0.0281 budget, so it loses by 0.125, and the free-fit oracle
#   said 0.0051. Thirty times the oracle is realisability, not capacity.
#
#   The checkpoint says which kind. The trained head's static background is UNIFORM to
#   four decimal places (entropy 10.395 of 10.397), so it holds no unigram prior and
#   pays to re-derive one from h at every position; and its components are NOT collapsed
#   (mean |cos| 0.037 to 0.171 across the three axes). So --sch-nfh-smooth attacks the
#   measured failure and anything aimed at collapse is already ruled out. --with-fixes
#   runs that pair.
#
#   VOCAB=32768 bash scripts/c14_nfh.sh 8                # the cheap screen, ~5x less
#   bash scripts/c14_nfh.sh 8                            # the headline, V=131,072
#   RANKS="1 32 64" bash scripts/c14_nfh.sh 8
#   bash scripts/c14_nfh.sh --with-controls --with-baselines 8
# ============================================================================
set -o pipefail

FORCE=0
SEEDS=1
RUN_MAIN=1
RUN_GLOBAL=0
RUN_CONTROLS=0
RUN_FIXES=0
RUN_BASELINES=0
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)          FORCE=1; shift ;;
        --seeds)          SEEDS="$2"; shift 2 ;;
        --with-controls)  RUN_CONTROLS=1; shift ;;
        --with-fixes)     RUN_FIXES=1; shift ;;
        --fixes-only)     RUN_MAIN=0; RUN_GLOBAL=0; RUN_FIXES=1; shift ;;
        --with-baselines) RUN_BASELINES=1; shift ;;
        --with-global)    RUN_GLOBAL=1; shift ;;
        --global-only)    RUN_MAIN=0; RUN_GLOBAL=1; shift ;;
        --controls-only)  RUN_MAIN=0; RUN_GLOBAL=0; RUN_CONTROLS=1; shift ;;
        [0-9]*)           DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--seeds N] [--with-controls] [--with-baselines]"
           echo "       [--with-fixes] [--fixes-only]"
           echo "       [--no-global] [--global-only] [--controls-only] [DEPTH ...]"
           exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(8)

# CHOOSING THE VOCABULARY. head/dense = R(1 + sum_g K_g)/V with sum_g K_g = G V^(1/G),
# so 0.0947x at V=32,768 against 0.0393x at V=131,072: independent of d, and the method
# gets relatively cheaper as V grows. Head share moves the same way, 35.2% against 68.4%
# at depth 8. Both make V=32,768 the HARSHER setting on cost, budget +0.0282 bpb against
# +0.0865. Whether it is harsher or kinder on quality is unmeasured: the offline numbers
# that once claimed otherwise were taken on a byte-level token stream and are retracted
# (LEARNINGS, "RETRACTED: the V=32,768 offline numbers"). Runs there are about 5x cheaper
# and the dense legs exist, so it stays the screen; the headline belongs at 131,072.
VOCAB="${VOCAB:-131072}"
# R=1 is not an optional extra: it is the LightRNN corner and the ablation the whole
# claim rests on, so it runs by default alongside the headline rank.
RANKS="${RANKS:-1 32}"
# Axes must TILE the padded vocabulary. These literals are the V=131,072 values, and
# they are literals only so the arm extractor in tests/test_code_head.py can read
# them; the guard below recomputes them from VOCAB and replaces them if they do not
# tile it, so VOCAB=32768 cannot silently keep 131,072's factorisation. An explicit
# DIMS that DOES tile the vocabulary is respected, since 128,32,32 is as legal as
# 64,64,32 and choosing between them is the point of the axis sweep.
DIMS="${DIMS:-64,64,32}"
DIMS2="${DIMS2:-512,256}"
# The measured dense slope AT THIS VOCABULARY. A slope carried over from another
# configuration is an assumption wearing the clothes of a measurement, and one of
# those already flipped two conclusions in this project (LEARNINGS, "The V=131k
# break-even was scored against the wrong dense curve").
SLOPE="${SLOPE:-0.186}"
if [ "$VOCAB" -lt 131072 ] && [ "$SLOPE" = "0.186" ]; then SLOPE=0.169; fi
tiles() { python3 -c "
import math, sys
sys.exit(0 if math.prod(int(x) for x in '$1'.split(',')) == $VOCAB else 1)"; }
derive_dims() { python3 -c "
import math
e = int(math.log2($VOCAB)); g = $1
print(','.join(str(2**k) for k in sorted(
    (e//g + (1 if i < e%g else 0) for i in range(g)), reverse=True)))"; }
for _v in DIMS:3 DIMS2:2; do
    _name="${_v%%:*}"; _g="${_v##*:}"
    if ! tiles "${!_name}"; then
        _new=$(derive_dims "$_g")
        echo "  [axes] ${_name}=${!_name} does not tile V=${VOCAB}; using ${_new}"
        printf -v "$_name" '%s' "$_new"
    fi
done
GLOBAL_M="${GLOBAL_M:-256}"
SMOOTH="${SMOOTH:-0}"
PERM_PATH="${PERM_PATH:-perms/nfh_g3_v${VOCAB}.pt}"
# A plain string, not an array: scripts/../tests extract arms by reading these
# assignments, and "${ARR[@]}" is not something that extractor can expand.
PERM_FLAGS="--sch-nfh-perm file --sch-nfh-perm-path $PERM_PATH"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/c14_nfh}"
RANK_CONTEXTS="${RANK_CONTEXTS:-16384}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
# Cost-matched plain low rank, for --with-baselines. The repo's rule is to DERIVE a
# matched control and never type one, because a hand-matched rank is right exactly
# once and then silently compares two budgets the moment R or the axes move. It is
# written literally here only so the arm extractor in tests/test_code_head.py can
# read it, and the baselines block below recomputes it and refuses to run if the two
# have drifted. The value is d*R*(1+sum_g K_g)/(d+V) at depth 8.
LOWRANK_C="${LOWRANK_C:-20}"

if [ "$VOCAB" -eq 131072 ]; then
    TOK="${TOKENIZER_DIR:-${TOKENIZER_DIR_131K:-tokenizer_131k}}"
else
    TOK="${TOKENIZER_DIR:-tokenizer}"
fi
CHECK_DATA_DIR="${DATA_DIR:-data}"
if [ -L "$CHECK_DATA_DIR" ] && [ ! -e "$CHECK_DATA_DIR" ]; then
    echo "  [data] removing dangling symlink at $CHECK_DATA_DIR"
    rm "$CHECK_DATA_DIR"
fi
mkdir -p "$CHECK_DATA_DIR"
if [ ! -f "$CHECK_DATA_DIR/shard_06542.parquet" ] || [ ! -f "$CHECK_DATA_DIR/shard_00000.parquet" ]; then
    echo "  [data] initial data shards missing in $CHECK_DATA_DIR; downloading..."
    python3 -m nanochat.dataset -n "${MAX_SHARDS:-8}" --data-dir "$CHECK_DATA_DIR"
fi

if ! python3 -m scripts.ensure_tokenizer --vocab-size "$VOCAB" --tokenizer-dir "$TOK" \
        --data-dir "$CHECK_DATA_DIR" ${MAX_SHARDS:+--max-shards "$MAX_SHARDS"}; then
    echo "could not prepare the tokenizer at '${TOK}'; nothing was run."
    exit 1
fi
mkdir -p "$OUT_BASE"

# The assignment decides which words share a code cell and is worth 3.3x the
# structural error offline, so a silent fallback to token-id order would turn the
# headline arm into its own control and the sweep would look like it ran. Build it if
# we are told where from, and otherwise stop.
#
#   PERM_CKPT   a dense .pt to fit the assignment from. A full checkpoint, a bare
#               (V, d) head, or the {"acts","lm_head"} payload dump_head_acts writes.
#   PERM_ACTS   optional activations that order the leaf axis by predictability;
#               worth 10-20% of the reconstruction error, free at run time. Defaults
#               to PERM_CKPT, which already carries them if it is an acts payload.
#   ALLOW_NO_PERM=1  run on token-id order on purpose. It is a control, not a default.
if [ ! -f "$PERM_PATH" ]; then
    if [ -n "${PERM_CKPT:-}" ]; then
        echo "  [perm] ${PERM_PATH} missing; fitting it from ${PERM_CKPT}"
        mkdir -p "$(dirname "$PERM_PATH")"
        if ! python3 -m scripts.build_vocab_permutation --mode nested \
                --dims "$DIMS" --vocab-size "$VOCAB" --checkpoint "$PERM_CKPT" \
                --acts "${PERM_ACTS:-$PERM_CKPT}" --iters "${PERM_ITERS:-12}" \
                --out "$PERM_PATH"; then
            echo "  [perm] could not build ${PERM_PATH}; nothing was run."
            exit 1
        fi
    elif [ "${ALLOW_NO_PERM:-0}" = "1" ]; then
        echo "  [perm] ALLOW_NO_PERM=1: running on token-id order. This is the control arm."
        PERM_FLAGS="--sch-nfh-perm none"
    else
        echo ""
        echo "  ${PERM_PATH} does not exist, and without it every arm would silently"
        echo "  measure token-id order instead of the fitted assignment. Nothing was run."
        echo ""
        echo "  Point the sweep at any V=${VOCAB} DENSE checkpoint and it will fit one:"
        echo "    PERM_CKPT=<dense model_*.pt | head .pt | acts .pt> \\"
        echo "        VOCAB=${VOCAB} bash $0 ${DEPTHS[*]}"
        echo ""
        echo "  Or build it yourself:"
        echo "    python -m scripts.build_vocab_permutation --mode nested \\"
        echo "        --dims ${DIMS} --vocab-size ${VOCAB} --checkpoint <dense .pt> \\"
        echo "        --acts <acts .pt> --out ${PERM_PATH}"
        echo ""
        echo "  Or ALLOW_NO_PERM=1 to run the token-id control on purpose."
        exit 1
    fi
fi

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

for DEPTH in "${DEPTHS[@]}"; do

MODEL_DIM=$(( ((DEPTH * ASPECT_RATIO + 127) / 128) * 128 ))
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c14_d${DEPTH}.log}"
STATE="${OUT_BASE}/c14_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# The DENSE arm's Chinchilla budget, pinned on every arm. This head has 25x fewer
# parameters than a dense softmax, so on base_train's default sizing it would draw a
# much smaller budget and the comparison would be confounded by data rather than
# architecture. Same argument as OPEN_QUESTIONS Q10 makes for Monarch.
TARGET_TOKENS="${TARGET_TOKENS:-$(python3 -m scripts.code_head_budget --depth "$DEPTH" --ratio "${RATIO:-10.5}" --tokenizer-dir "$TOK")}"

COMMON="--device-batch-size $DEVICE_BATCH_SIZE --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir $TOK \
  --sequence-len ${SEQ_LEN:-2048} --target-tokens $TARGET_TOKENS \
  --target-param-data-ratio -1 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 \
  --save-every 200 --eval-every -1 \
  --sch-decile-metrics 1 --sch-eval-steps ${EVAL_STEPS:-100}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"
PROBE="--sch-phi-dtype fp32 --sch-rank-probe $RANK_CONTEXTS"
NFH="--models base --use-code-head 1 --sch-head-type nfh $PROBE"

run() {
    local tag="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "SKIP  $t (already completed)"; continue; fi
        echo ""
        echo "--- $t  (depth $DEPTH, V=${VOCAB}, device-batch ${DEVICE_BATCH_SIZE}) ---"
        local dir="${OUT_BASE}/d${DEPTH}/${t}"
        [ "$FORCE" -eq 1 ] && rm -rf "$dir"
        if bash scripts/research_sweep.sh $COMMON --out-dir "$dir" --seed "$s" \
               "$@" "$DEPTH" 2>&1 | tee -a "$LOGFILE"; then
            mark_done "$t"; echo "OK    $t"
        else
            echo "FAIL  $t (will retry on the next invocation)"
        fi
    done
}

echo "============================================================"
echo "  C14: nonnegative factorized head, V=${VOCAB}, depth ${DEPTH}, d=${MODEL_DIM}"
echo "  cp axes ${DIMS}   ranks ${RANKS}   assignment ${PERM_FLAGS}"
printf "  %-16s %-11s %-11s %-13s %s\n" "arm" "head MACs" "vs dense" "total FLOPs" "break-even bpb"
for R in $RANKS; do
    python3 -c "
import math
V,d,R = $VOCAB,$MODEL_DIM,$R
K = sum(int(x) for x in '$DIMS'.split(','))
head = d*R*(1+K); dense = V*d
share = V/(V + 12*d*$DEPTH); ratio = (1-share) + share*head/dense
print(f'  {\"NFH_cp_R\"+str(R):<16} {head/1e6:8.2f}M  {head/dense:9.4f}x  {ratio:11.4f}x  '
      f'{-math.log10(ratio)*$SLOPE:+.4f}')"
done
if [ "$RUN_GLOBAL" -eq 1 ]; then
    python3 -c "
import math
V,d,m = $VOCAB,$MODEL_DIM,$GLOBAL_M
head = d*m + V*m//32768; dense = V*d
share = V/(V + 12*d*$DEPTH); ratio = (1-share) + share*head/dense
print(f'  {\"NFH_global_m\"+str(m):<16} {head/1e6:8.2f}M  {head/dense:9.4f}x  {ratio:11.4f}x  '
      f'{-math.log10(ratio)*$SLOPE:+.4f}')"
fi
echo "  (break-even uses the measured dense slope at V=${VOCAB}: ${SLOPE} bpb/decade)"
echo "  target tokens ${TARGET_TOKENS}   device-batch ${DEVICE_BATCH_SIZE}"
echo "============================================================"

if [ "$RUN_MAIN" -eq 1 ]; then
    for R in $RANKS; do
        run "NFH_cp_R${R}" $NFH --sch-nfh-mode cp --sch-nfh-dims "$DIMS" \
            --sch-nfh-rank "$R" $PERM_FLAGS --sch-nfh-smooth "$SMOOTH"
    done
fi
# The two arms that attack the MEASURED failure of the first d8 sweep, where R=32 came
# in 0.1529 bpb above dense against a +0.0281 budget while its free-fit oracle said
# 0.0051. Thirty times the oracle is realisability, not capacity, so both of these
# widen the path from h to the factors rather than adding components.
#   smooth  a word no component points at falls to the background product, about
#           (1/32)^3 at these axes against a true unigram of 1e-4 to 1e-3. One gather
#           per token buys the unigram back, and 0.1529 bpb is 0.501 nats, which is
#           the right size for that gap.
#   mlp     the map itself: 3,104 factor parameters out of a 512-dimensional h is a
#           6x compression, and this is the only knob that relieves it. Costs 16% more
#           head FLOPs, moving the budget +0.0281 -> +0.0271.
if [ "$RUN_FIXES" -eq 1 ]; then
    for R in $RANKS; do
        [ "$R" = "1" ] && continue
        run "NFH_cp_R${R}_smooth" $NFH --sch-nfh-mode cp --sch-nfh-dims "$DIMS" \
            --sch-nfh-rank "$R" $PERM_FLAGS --sch-nfh-smooth 1
        run "NFH_cp_R${R}_mlp" $NFH --sch-nfh-mode cp --sch-nfh-dims "$DIMS" \
            --sch-nfh-rank "$R" $PERM_FLAGS --sch-nfh-g-type mlp
        run "NFH_cp_R${R}_smooth_mlp" $NFH --sch-nfh-mode cp --sch-nfh-dims "$DIMS" \
            --sch-nfh-rank "$R" $PERM_FLAGS --sch-nfh-smooth 1 --sch-nfh-g-type mlp
    done
fi

# The hedge. Its per-context degrees of freedom sit BELOW d, so its offline oracle
# is close to a realisability guarantee rather than an upper bound, and it is another
# 20x cheaper again.
if [ "$RUN_GLOBAL" -eq 1 ]; then
    run "NFH_global_m${GLOBAL_M}" $NFH --sch-nfh-mode global --sch-nfh-rank "$GLOBAL_M"
fi

if [ "$RUN_CONTROLS" -eq 1 ]; then
    # Is the assignment load-bearing, or would any bijection do? Offline the fitted
    # one is worth 3.3x over token-id order and 7x over random.
    for R in $RANKS; do
        [ "$R" = "1" ] && continue
        run "NFH_cp_R${R}_permnone"   $NFH --sch-nfh-mode cp --sch-nfh-dims "$DIMS" \
            --sch-nfh-rank "$R" --sch-nfh-perm none
        run "NFH_cp_R${R}_permrandom" $NFH --sch-nfh-mode cp --sch-nfh-dims "$DIMS" \
            --sch-nfh-rank "$R" --sch-nfh-perm random
        # Factorisation depth: 2 axes cost 4.8x more and the oracle says they buy 3x
        # less error. Whether that trade survives training is a separate question.
        run "NFH_cp_g2_R${R}" $NFH --sch-nfh-mode cp --sch-nfh-dims "$DIMS2" \
            --sch-nfh-rank "$R" --sch-nfh-perm none
    done
fi

if [ "$RUN_BASELINES" -eq 1 ]; then
    # The control has to cost what the headline arm costs. Recompute and refuse to
    # run on a stale value rather than quietly comparing two different budgets.
    HR=$(echo $RANKS | tr ' ' '\n' | tail -1)
    DERIVED=$(python3 -c "
V,d,R = $VOCAB,$MODEL_DIM,$HR
K = sum(int(x) for x in '$DIMS'.split(','))
print(max(1, round(d*R*(1+K)/(d+V))))")
    if [ "$LOWRANK_C" != "$DERIVED" ]; then
        echo "LOWRANK_C=${LOWRANK_C} but the cost-matched rank at depth ${DEPTH}," \
             "R=${HR}, axes ${DIMS} is ${DERIVED}. Re-run with LOWRANK_C=${DERIVED}."
        exit 1
    fi
    run "LOWRANK_c${LOWRANK_C}" --models base --use-code-head 1 \
        --sch-phi-mode learned --sch-max-m "$LOWRANK_C" $PROBE --sch-bias 1
    # Both of these are implemented and have never been run. The offline work says
    # hsoftmax loses by 0.568 bpb; running it is what makes that a measurement in the
    # paper rather than an extrapolation from a distillation oracle.
    run "HSOFTMAX" --models base --use-code-head 1 --sch-head-type hsoftmax $PROBE
    run "TIERED"   --models base --use-code-head 1 --sch-head-type tiered \
        --sch-tier-bounds "${TIER_BOUNDS:-1024,4096,16384,65536}" \
        --sch-tier-caps "${TIER_CAPS:-512,512,511,480,18}" \
        --sch-tier-order freq $PROBE
    run "DENSE" --models base $PROBE
fi

done

echo ""
echo "============================================================"
echo "  C14 complete."
echo ""
echo "    python -m scripts.sweep_report ${OUT_BASE}/d<DEPTH>"
echo ""
echo "  Read it in this order, and stop at the first one that fails."
echo ""
echo "    1. NFH_cp_R32 against NFH_cp_R1. R=1 IS LightRNN. If the mixture is worth"
echo "       nothing here the central claim is wrong and nothing else matters."
echo "       Offline the gap is 0.2829 -> 0.0097 bpb, a factor of 29."
echo "    2. NFH_cp_R32 bpb against dense at the same total FLOPs. The budget at d8"
echo "       is +0.0865 bpb and the oracle spends 0.0097 of it on the family. What is"
echo "       left is realisability: whether a 512-dimensional h can actually produce"
echo "       5,152 factor parameters per token. That is the number this run exists for."
echo "    3. NFH_global_m256. The hedge, at 0.317x total FLOPs. If cp fails on"
echo "       realisability and global does not, the paper is global and the finding is"
echo "       sharper rather than weaker."
echo "    4. ms/step. The entire argument is bandwidth: the widest activation is"
echo "       N x 5120 against dense's N x 131,072. If this is not at least as fast as"
echo "       dense per step then something is wrong with the implementation, not with"
echo "       the idea, because there is no gather and no custom kernel anywhere in it."
echo ""
echo "  d8 V=131,072 reference points, same budget:"
echo "    dense                          0.902908 at 98.13M MACs/token"
echo "    LOWRANK_M259                   0.918721 at 65.03M"
echo "    best structured (c11)          0.911994 at 65.03M"
echo "    NFH_cp_R32                     predicted 0.90-0.92 at 33.66M"
echo "============================================================"
