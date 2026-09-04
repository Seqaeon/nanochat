#!/usr/bin/env bash
# ============================================================================
# C05: Phase 5. The five directions that survive the c00 post-mortem.
#
# WHY THIS SWEEP EXISTS
#   c00 did not fail because the theory was wrong. The rank gate passed exactly
#   (order 1 -> 15 = B, order 2 -> 120 = M, dense -> 256 = d). It failed because
#   the *instantiation* was wrong on two counts, both of which this sweep fixes.
#
#   1. THE PRIZE WAS MIS-SIZED. c00 ran at V=32768 where the head is 9% of model
#      FLOPs at depth 20. Current models use 128k (Llama 3) to 256k (Gemma). At
#      V=131k the head is 50.7% of FLOPs at depth 12 and 28.3% at depth 20. Set
#      VOCAB_SIZE to the size you actually care about before reading anything
#      here as a Pareto claim.
#
#   2. THE SAVING WAS MIS-TARGETED. Freezing Phi removes the weight gradient,
#      worth 2 of 6 FLOPs per MAC, so one third of the head. That was the whole
#      prize c00 chased. Making the head structurally cheaper is worth most of
#      the head. Groups (product) and (monarch) chase the larger target.
#
#   And the measured wall from c00 is 0.716 bpb: at M=120, same g, same budget,
#   a LEARNED Phi scores 1.2281 and the frozen monomial Phi scores 1.9437. Three
#   capacity settings (k3 linear, k3 MLP, k4 MLP) land within 0.044 bpb of each
#   other, so it is not a rank wall. Every group below attacks that number from a
#   different side.
#
# THE FIVE GROUPS
#   (product)  K-ary product codes. Read PROD_g2_K512 against PROD_g16_K64:
#              same M=1024, 8x apart in gather cost. A binary digit contributes ONE column to Phi,
#              so B digits buy M = B at order 1 and the only way to widen is
#              interaction order, which multiplies columns inside the lattice the
#              same B digits already generate. That is the c00 plateau. A K-ary
#              digit contributes K columns, so g digits give M = g*K at order 1
#              with no interactions. Phi is one-hot per group, which turns the
#              V x M product into a gather and add costing V*g instead of V*M.
#              Screened on the c00 dense head: binary order-2 captures 1.79% of
#              its logit energy at M=120, product captures 21.93% at M=128 and
#              keeps growing (46.52% at M=2048) where monomials saturate.
#
#   (mixture)  Union of subspaces. c00's mixture computed ALL components (cost
#              K * 4VM) and they SHARED one Phi, so it could only reweight a
#              single subspace. Inverted here: per-component Phi, top-1 routing.
#              Reach becomes the union of K subspaces at the per-token cost of
#              one. After "the capital of France is" the useful logit directions
#              are place names; after "def foo(" they are code tokens.
#
#   (monarch)  Attack the same FLOP bill from the other side: keep the map fully
#              LEARNED so there is no alignment question at all, and make it
#              cheap structurally. Two block-diagonal factors with a transpose
#              between, costing d*M + V*m1 instead of V*d.
#
#   (tree)     Hierarchical softmax. Exact cross-entropy at O(d log V). It has
#              been implemented in this repo since c00 and never run. One arm to
#              place the point, because a reviewer will ask.
#
#   (free)     The two things c00 skipped that cost almost nothing. A per-token
#              bias is 2V FLOPs and c00 ran every code arm without it. Whitening
#              Phi is provably a reparameterisation for a linear g, so the
#              function class is IDENTICAL and any bpb movement is purely an
#              optimisation effect. That single run separates "wrong subspace"
#              from "badly conditioned" for good.
#
# THE BASELINE THAT MATTERS IS NOT DENSE
#   CTRL_learned_W scored 1.2281 at 66% of dense FLOPs in c00. Every arm here is
#   competing against that, not against the softmax. An arm that beats dense on
#   FLOPs but loses to a plain learned rank-M head has shown nothing.
#
# WHY THE PRODUCT ARMS RUN --sch-product-impl dense
#   MEASURED, not predicted: the gather path is 4.76x SLOWER than the matmul it
#   replaces (CPU, N=2048, V=32768, g=8, K=64), and the matmul does 64x more
#   arithmetic. On a GPU it is worse: the backward of index_select is an
#   index_add scattering V values into K slots per position, which at V=32768
#   and K=64 is 512-way atomic contention per slot. A first attempt at this
#   sweep ran at 3.36 s/step and 0.23 MFU because of it.
#
#   So the quality question and the cost question are separated here. The dense
#   impl materialises the one-hot Phi (33 MB at V=32768 M=512) and runs a normal
#   GEMM, costing 4*V*M like any other frozen Phi. That makes every product arm
#   COST-MATCHED to a monomial or random-binary arm at the same M, which is
#   exactly the comparison that answers "does a K-ary partition span a better
#   subspace than a binary one". The V*g cost claim needs a fused kernel and is
#   tracked separately in OPEN_QUESTIONS Q8; the (kernel) group below measures
#   the gather so the gap is on the record rather than assumed.
#
# WALL CLOCK IS PART OF THE RESULT, AND THE FLOP COLUMN WILL LIE TO YOU
#   Both heads write the same N x V logit tensor. The dense head is compute
#   bound at 768 FLOP/byte, so removing its compute lands on that write and not
#   on zero. H100 roofline at V=131072: dense 13.3 ms, a FUSED product head
#   5.1 ms. The FLOP ratio is 96x and the achievable ratio is 2.6x.
#
#   Worse, product_gather as currently written is the UNFUSED version: one
#   index_select per group, each materialising a full N x V tensor, so about 2g
#   passes over the output. Roofline puts that at 82 ms, 6x SLOWER than dense,
#   while sch_results.csv happily reports a 96x FLOP reduction.
#
#   So read the head timing columns before believing any FLOP ratio here, and
#   fix the kernel first if the PROD_* arms regress on step time. torch.compile
#   may fuse the chain on its own; Triton is the fallback. Same lesson as
#   OPEN_QUESTIONS Q6 (the MST d_h=32 kernel penalty) and Q11.
#
#   bash scripts/c05_sch_phase5_alternatives.sh                     # depth 8, all groups
#   bash scripts/c05_sch_phase5_alternatives.sh --group product 8   # one group
#   bash scripts/c05_sch_phase5_alternatives.sh --seeds 3 8         # with error bars
#   VOCAB_SIZE=131072 bash scripts/c05_sch_phase5_alternatives.sh 12
#   PROXY_CKPT=out/c00_sch_phase0/d4/DENSE_softmax_s1/depth_4/ckpt_base \
#       bash scripts/c05_sch_phase5_alternatives.sh --group product 8
# ============================================================================
set -o pipefail

FORCE=0
RUN_GROUPS="baseline product mixture monarch tree free"   # "kernel" is opt-in: it is slow on purpose
SEEDS=1
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)  FORCE=1; shift ;;
        --group)  RUN_GROUPS="$2"; shift 2 ;;
        --seeds)  SEEDS="$2"; shift 2 ;;
        [0-9]*)   DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--group G] [--seeds N] [DEPTH ...]"; exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=("${DEPTH:-8}")

ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/c05_sch_phase5}"
RANK_CONTEXTS="${RANK_CONTEXTS:-16384}"
PROXY_CKPT="${PROXY_CKPT:-}"
CODE_DIR="${CODE_DIR:-${OUT_BASE}/codes}"
mkdir -p "$OUT_BASE" "$CODE_DIR"

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

has() { echo " $RUN_GROUPS " | grep -q " $1 "; }

# ==== per-depth body ========================================================
for DEPTH in "${DEPTHS[@]}"; do

MODEL_DIM=$(( ((DEPTH * ASPECT_RATIO + 127) / 128) * 128 ))
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c05_d${DEPTH}.log}"
STATE="${OUT_BASE}/c05_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# Same pinning as c00 and for the same reason: base_train sizes the token budget
# from head parameters, and these heads have between 17x fewer and 2x more of
# them than dense. Per-arm Chinchilla budgeting would hand each arm a different
# budget and confound every comparison in the sweep. Compute the DENSE budget
# once and pin it on every arm.
TARGET_TOKENS="${TARGET_TOKENS:-$(python3 -m scripts.code_head_budget --depth "$DEPTH" --ratio "${RATIO:-10.5}" --tokenizer-dir "${TOKENIZER_DIR:-tokenizer}")}"

COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len ${SEQ_LEN:-2048} --target-tokens $TARGET_TOKENS \
  --target-param-data-ratio -1 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 \
  --save-every 200 --eval-every -1 \
  --sch-decile-metrics 1 --sch-eval-steps ${EVAL_STEPS:-100}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

PROBE="--sch-phi-dtype fp32 --sch-rank-probe $RANK_CONTEXTS"

run() {                                   # run <tag> <depth> <flags...>
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "SKIP  $t (already completed)"; continue; fi
        echo ""
        echo "--- $t  (depth $depth) ---"
        local dir="${OUT_BASE}/d${depth}/${t}"
        [ "$FORCE" -eq 1 ] && rm -rf "$dir"
        if bash scripts/research_sweep.sh $COMMON --out-dir "$dir" --seed "$s" \
               "$@" "$depth" 2>&1 | tee -a "$LOGFILE"; then
            mark_done "$t"; echo "OK    $t"
        else
            echo "FAIL  $t (will retry on the next invocation)"
        fi
    done
}

echo "============================================================"
echo "  C05 Phase 5: the alternatives that survive the c00 post-mortem"
echo "  depth ${DEPTH}   d=${MODEL_DIM}   seeds ${SEEDS}"
echo "  target tokens ${TARGET_TOKENS}   rank contexts ${RANK_CONTEXTS}"
echo "  groups: ${RUN_GROUPS}"
echo "  out ${OUT_BASE}"
echo "============================================================"

# ---------------------------------------------------------------- baseline
# Two reference points, both required. Dense sets the FLOPs-bpb curve. The
# learned rank-M head is the arm every structured head actually has to beat,
# and c00 already put it at 1.2281 for 66% of dense FLOPs.
if has baseline; then
    echo ""; echo "### BASELINE: dense softmax, and the learned rank-M head to beat"
    run BASE_dense "$DEPTH" --models base --sch-rank-probe $RANK_CONTEXTS
    run BASE_learned_W "$DEPTH" --models base --use-code-head 1 \
        --sch-phi-mode learned --sch-max-m 120 $PROBE
    run BASE_code_order2 "$DEPTH" --models base --use-code-head 1 \
        --sch-order 2 $PROBE
fi

# ---------------------------------------------------------------- product
# The leading candidate. Sweep the alphabet at roughly fixed M to separate "more
# basis functions" from "better-shaped basis functions", then push M up where
# the monomial ladder could not follow.
#
# The (fit) arms need an assignment fitted to a proxy model's embedding. Set
# PROXY_CKPT to a trained dense checkpoint; without it those arms are skipped
# rather than silently falling back to the hash control, because a silent
# fallback would look like a fitted result and would not be one.
if has product; then
    echo ""; echo "### PRODUCT: K-ary codes. M = g*K at order 1, cost V*g not V*M."
    # A g-digit code over K symbols has K^g cells and needs at least V of them, so
    # (g, K) pairs are not free to choose. At V=32768 the smallest g at K=64 is 3,
    # and g must divide the model width for the fitted arms, so the K=64 sweep
    # starts at g=4. g=2 (the LightRNN corner) needs K >= 512 to stay legal
    # through V=262144. An illegal pair is refused at construction rather than
    # trained with colliding tokens, which would put a hard floor under the loss.
    run PROD_g2_K512  "$DEPTH" --models base --use-code-head 1 --sch-phi-mode product \
        --sch-product-groups 2  --sch-product-codebook 512 --sch-bias 1 $PROBE
    run PROD_g4_K64   "$DEPTH" --models base --use-code-head 1 --sch-phi-mode product \
        --sch-product-groups 4  --sch-product-codebook 64  --sch-bias 1 $PROBE
    run PROD_g8_K64   "$DEPTH" --models base --use-code-head 1 --sch-phi-mode product \
        --sch-product-groups 8  --sch-product-codebook 64  --sch-bias 1 $PROBE
    # Matched to PROD_g2_K512 at M=1024 with 8x the gather cost, so the pair
    # separates "many cheap digits" from "few rich digits" at fixed width.
#    run PROD_g16_K64  "$DEPTH" --models base --use-code-head 1 --sch-phi-mode product \
#        --sch-product-groups 16 --sch-product-codebook 64  --sch-bias 1 $PROBE
#    run PROD_g8_K256  "$DEPTH" --models base --use-code-head 1 --sch-phi-mode product \
#        --sch-product-groups 8  --sch-product-codebook 256 --sch-bias 1 $PROBE
#    # The MLP-g slice, for the same reason c00 needed one: a linear g caps the
#    # rank at min(M, d), so g=8 K=256 (M=2048) and g=8 K=64 (M=512) are
#    # rank-identical at d<=512 and the alphabet sweep would fake saturation.
#    run PROD_g8_K256_mlp "$DEPTH" --models base --use-code-head 1 --sch-phi-mode product \
#        --sch-product-groups 8 --sch-product-codebook 256 --sch-bias 1 \
#        --sch-g-type mlp --sch-g-hidden "$MODEL_DIM" $PROBE
#    # The null control. If a random assignment matches a fitted one, the code is
#    # not doing the work and only the width is.
#    run PROD_g8_K64_random "$DEPTH" --models base --use-code-head 1 --sch-phi-mode product \
#        --sch-product-groups 8 --sch-product-codebook 64 --sch-product-source random \
#        --sch-bias 1 $PROBE
    # COST-MATCHED CONTROLS at M=512. Under --sch-product-impl dense all three
    # cost exactly 4*V*M, so bpb differences are attributable to the SUBSPACE and
    # to nothing else. This is the comparison the whole product hypothesis rests
    # on: a K-ary partition against a binary monomial one at identical width and
    # identical price.
    run PROD_ctrl_monomial_M512 "$DEPTH" --models base --use-code-head 1 \
        --sch-order 3 --sch-max-m 512 --sch-bias 1 $PROBE
    run PROD_ctrl_randbin_M512  "$DEPTH" --models base --use-code-head 1 \
        --sch-phi-mode random_binary --sch-max-m 512 --sch-bias 1 $PROBE

    if [ -n "$PROXY_CKPT" ]; then
        for GK in "4 64" "8 64" "8 256"; do
            set -- $GK; G=$1; K=$2
            CODEF="${CODE_DIR}/prod_g${G}_K${K}_d${DEPTH}.pt"
            if [ ! -f "$CODEF" ]; then
                echo "fitting product code g=${G} K=${K} from ${PROXY_CKPT}"
                python3 -m scripts.code_assign --mode product \
                    --from-checkpoint "$PROXY_CKPT" --model-tag "${PROXY_TAG:-base}" \
                    --output-embedding --product-groups "$G" --product-codebook "$K" \
                    --tokenizer-dir "${TOKENIZER_DIR:-tokenizer}" --out "$CODEF" || continue
            fi
            run "PROD_fit_g${G}_K${K}" "$DEPTH" --models base --use-code-head 1 \
                --sch-phi-mode product --sch-product-groups "$G" --sch-product-codebook "$K" \
                --sch-product-source file --sch-code-path "$CODEF" --sch-bias 1 $PROBE
        done
    else
        echo ""
        echo "SKIP  PROD_fit_* : set PROXY_CKPT to a trained dense checkpoint to fit"
        echo "      the assignment to its output embedding. The hash arms above are"
        echo "      the control, not a substitute."
    fi
fi

# ---------------------------------------------------------------- kernel
# The gather path, run on purpose so the cost gap is measured rather than
# asserted. Expect it to be SLOW: it exists to put a number on Q8, not to
# produce a bpb. Identical model to PROD_g8_K64, so any bpb difference between
# the two is a bug and the step-time difference is the whole point.
if has kernel; then
    echo ""; echo "### KERNEL: the V*g gather path. Expected slower. Read step time, not bpb."
    run KERN_g8_K64_gather "$DEPTH" --models base --use-code-head 1 --sch-phi-mode product \
        --sch-product-groups 8 --sch-product-codebook 64 --sch-product-impl gather \
        --sch-bias 1 $PROBE
fi

# ---------------------------------------------------------------- mixture
# Union of subspaces. sch-max-m must stay below the full order-2 width or every
# component draws the SAME monomial subset and the arm is a no-op; the head
# refuses that configuration rather than running it.
if has mixture; then
    echo ""; echo "### MIXTURE: per-component Phi, top-1 routing. Cost of one, reach of K."
    # Sparse routing needs two things that dense mixtures do not, both now on by
    # default. The router cannot start symmetric: with zero weights every token's
    # logits tie and topk breaks the tie by index, so component 0 takes every
    # token and 1..K-1 are permanently dead. That failed as a DDP error
    # ("Parameter indices which did not receive grad: 31 32 33"). And --sch-mixture-aux
    # (Switch load balancing, default 0.01) keeps them alive afterwards; without
    # it top-1 concentrates on whichever component wins early and the arm pays
    # for K subspaces while using one.
    #
    # order 3 TRUNCATED to M=120, not order 2. At B=15 the full order-2 width IS
    # 120, so every component would draw the same monomial set and the arm would
    # be a no-op; the head refuses that rather than running it. Truncating order
    # 3 (full width 575) to 120 gives each component a different 120-subset at
    # the same M as c00's order-2 arm, so the comparison stays matched.
    run MIX_k4_top1 "$DEPTH" --models base --use-code-head 1 --sch-order 3 --sch-max-m 120 \
        --sch-mixture 4 --sch-mixture-per-phi 1 --sch-mixture-topk 1 --sch-bias 1 $PROBE
    run MIX_k8_top1 "$DEPTH" --models base --use-code-head 1 --sch-order 3 --sch-max-m 120 \
        --sch-mixture 8 --sch-mixture-per-phi 1 --sch-mixture-topk 1 --sch-bias 1 $PROBE
    run MIX_k8_top2 "$DEPTH" --models base --use-code-head 1 --sch-order 3 --sch-max-m 120 \
        --sch-mixture 8 --sch-mixture-per-phi 1 --sch-mixture-topk 2 --sch-bias 1 $PROBE
    # The control that isolates the union from the mixing. Same K, same routing,
    # but ONE shared Phi: this is what c00's mixture already was.
    run MIX_k8_shared_phi "$DEPTH" --models base --use-code-head 1 --sch-order 3 --sch-max-m 120 \
        --sch-mixture 8 --sch-mixture-topk 1 --sch-bias 1 $PROBE
fi

# ---------------------------------------------------------------- monarch
# Fully learned, so alignment cannot be the explanation for anything it does.
# If Monarch lands near dense, the c00 deficit was the freezing and not the
# structure; if it lands near the code heads, structure is the problem and the
# whole family is in trouble. Either way it is the cleanest single arm here.
if has monarch; then
    echo ""; echo "### MONARCH: cost d*M + V*m1, every parameter trained"
    run MON_M256  "$DEPTH" --models base --use-code-head 1 --sch-head-type monarch \
        --sch-max-m 256  --sch-bias 1 $PROBE
    run MON_M1024 "$DEPTH" --models base --use-code-head 1 --sch-head-type monarch \
        --sch-max-m 1024 --sch-bias 1 $PROBE
    run MON_M1024_m1_8 "$DEPTH" --models base --use-code-head 1 --sch-head-type monarch \
        --sch-max-m 1024 --sch-monarch-m1 8 --sch-bias 1 $PROBE
fi

# ---------------------------------------------------------------- tree
# Exact cross-entropy at O(d log V). Implemented since c00 and never run. It
# cannot materialise a logit vector, so it carries no rank probe and no decile
# metrics; that is a property of the method, not a missing feature.
if has tree; then
    echo ""; echo "### TREE: Huffman hierarchical softmax, the classical log-V baseline"
    run TREE_hsoftmax "$DEPTH" --models base --use-code-head 1 --sch-head-type hsoftmax \
        --sch-decile-metrics 0 --sch-rank-probe 0
fi

# ---------------------------------------------------------------- free
# Two cheap arms that reinterpret every number c00 produced.
#
# BIAS: a per-token bias costs 2V FLOPs and c00 ran every code arm without one.
# WHITEN: for a linear g, Phi (Phi^T Phi)^-1/2 spans the same subspace, so the
#   function class is identical and the arm can ONLY move through optimisation.
#   A null result kills the conditioning hypothesis permanently; a large one
#   says c00 measured an optimisation failure and called it a capacity limit.
#   Measured cond(Phi^T Phi) after row normalisation: 396 at order 2, 3225 at
#   order 3. fp32 is mandatory here, as the eigendecomposition is done at build.
if has free; then
    echo ""; echo "### FREE: the per-token bias and the whitened Phi that c00 skipped"
    run FREE_order2_bias   "$DEPTH" --models base --use-code-head 1 --sch-order 2 \
        --sch-bias 1 $PROBE
    run FREE_order2_whiten "$DEPTH" --models base --use-code-head 1 --sch-order 2 \
        --sch-phi-whiten 1 $PROBE
    run FREE_order2_both   "$DEPTH" --models base --use-code-head 1 --sch-order 2 \
        --sch-bias 1 --sch-phi-whiten 1 $PROBE
    run FREE_order3_both   "$DEPTH" --models base --use-code-head 1 --sch-order 3 \
        --sch-bias 1 --sch-phi-whiten 1 $PROBE
fi

echo ""
echo "============================================================"
echo "  C05 depth ${DEPTH} complete."
echo ""
echo "  READ:  sch_results.csv, columns val_bpb, num_flops_per_token,"
echo "         rank_effective_rank, and the head timing columns."
echo ""
echo "  THE COMPARISON THAT DECIDES THIS: every arm against BASE_learned_W,"
echo "  not against BASE_dense. c00 put a plain learned rank-120 head at"
echo "  1.2281 bpb for 66% of dense FLOPs. An arm that beats dense on FLOPs"
echo "  but loses to that has shown nothing."
echo ""
echo "  PRODUCT: the arms are cost-matched (impl=dense, 4*V*M), so read them as"
echo "           a QUALITY comparison at fixed price. PROD_g8_K64 against"
echo "           PROD_ctrl_monomial_M512 and PROD_ctrl_randbin_M512 at M=512 is"
echo "           the whole hypothesis: same width, same FLOPs, different"
echo "           subspace. Then PROD_g8_K64_random: if it matches the hash and"
echo "           fitted arms, only the width matters and the code is decoration."
echo ""
echo "  KERNEL:  run --group kernel to put a number on the gather. It should"
echo "           match PROD_g8_K64 on bpb exactly and lose badly on step time."
echo "           That gap is OPEN_QUESTIONS Q8 and it is what stands between"
echo "           the arithmetic claim and a wall-clock claim."
echo ""
echo "  MIXTURE: check the router actually spread. MIX_k8_shared_phi is the"
echo "           control: same K, same routing, one shared Phi. If it matches"
echo "           MIX_k8_top1 then the union of subspaces bought nothing and"
echo "           only the log-sum-exp mixing mattered."
echo ""
echo "  MONARCH: the diagnostic arm. Fully learned, so if MON_M1024 lands near"
echo "           BASE_dense then c00's deficit was freezing, and if it lands"
echo "           near BASE_code_order2 the problem is structural and the whole"
echo "           cheap-head family is in doubt."
echo ""
echo "  WHITEN:  FREE_order2_whiten has an IDENTICAL function class to"
echo "           BASE_code_order2 for a linear g. If they differ, c00 measured"
echo "           an optimisation failure. If they match, conditioning is dead"
echo "           as an explanation and misalignment is what is left."
echo ""
echo "  WALL CLOCK: the product head's gather-add is memory bound. Check the"
echo "              head timing columns before quoting the FLOP ratio."
echo ""
echo "  Re-measure any arm afterwards with:"
echo "    python -m scripts.code_head_diagnostics --checkpoint-dir ${OUT_BASE}/d${DEPTH}/<TAG>"
echo "    python -m scripts.code_head_subspace --checkpoint <dense arm>/model_XXXX.pt"
echo "============================================================"

done
