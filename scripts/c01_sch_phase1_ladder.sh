#!/usr/bin/env bash
# ============================================================================
# C01: Phase 1 of the Structured Code Output Heads plan. The core experiment.
#      The interaction-order ladder, the redundancy axis, the full baseline
#      table, and the code-assignment arms.
#
# WHAT THIS SWEEP IS FOR
#   Phase 0 established that the rank bound is real and that we can measure it.
#   This sweep asks the question the paper actually answers: is a frozen
#   structured Phi better or worse than a learned dense W at matched effective
#   width, and where does quality saturate as the width grows.
#
# THE EXPANSION WIDTHS, so the results can be read against them
#   M = sum_{j<=k} C(B, j). The empirical head-rank threshold below which quality
#   degrades regardless of model size is about 1000 (Godey et al. 2024).
#
#     B=15 (minimal at V=32768):  k=1     15   k=2    120   k=3    575   k=4   1940
#     B=24                     :  k=1     24   k=2    300   k=3   2324
#     B=32                     :  k=1     32   k=2    528   k=3   5488
#     B=64                     :  k=1     64   k=2   2080   k=3  capped at V
#     B=17 (minimal at V=131072): k=1     17   k=2    153   k=3    833
#
#   So the threshold is crossed between order 3 and order 4 at B=15, and already
#   at order 2 for B=64.
#
# THE PREDICTION UNDER TEST: REDUNDANCY BEATS DEPTH
#   Rank can be raised two ways, by interaction ORDER or by code REDUNDANCY
#   (B > log2 V). B=64 at order 2 reaches M=2080 with nothing more than a B x B
#   head, which is far easier to implement than an order-3 or order-4 expansion.
#   If that matches or beats B=15 at order 4 (M=1940, similar width, much more
#   machinery), it is the quotable result of the paper and it makes the method
#   practical. That comparison runs FIRST, in the (redundancy) group, because it
#   is the go/no-go for the headline claim.
#
#   The plan's concrete prediction, worth writing down before the runs so the
#   result is a test rather than a reading: at V=131072, order-2 / B=17 (rank 153)
#   lands 5 to 15 percent worse than full softmax, order 3 (833) approaches
#   parity, and B=64 order 2 (2080) reaches parity. If order 2 at minimal B
#   already matches softmax, the theory OVER-predicts the problem and the framing
#   has to change. Finding that out early is the point of running it early.
#
# THE OBJECTIVE IS EXACT CROSS-ENTROPY ON EVERY ARM, INCLUDING ORDER 1
#   There is no per-bit BCE path in this implementation, deliberately. BCE over
#   bits IS the independence assumption the interaction expansion exists to
#   remove, so an order-2 head trained with it would be incoherent. It also
#   breaks outright for the redundancy arms: once B > log2 V, independent
#   Bernoullis put probability mass on codewords that correspond to no token, so
#   the (redundancy) and B in {24,32,64} rungs below would be silently
#   uninterpretable under BCE. Every arm computes logits = g(h) @ Phi^T and hands
#   them to cross-entropy over the real vocabulary.
#
#   That makes BASE_oda_order1 a STRONGER baseline than Oda et al.'s own
#   objective, not a weaker one: same rank-B model, exactly normalised. Say so in
#   the paper rather than letting a reviewer assume the comparison was rigged.
#   The cost argument survives too, and is the reason the exact loss is
#   affordable here: at M=120 and V=32768 the logit matmul is about 3.9M MACs per
#   token against the softmax's 16.8M, so order 2 is roughly 4x cheaper than the
#   softmax it replaces while being exact. Per-bit BCE only ever existed to dodge
#   the O(V) computation, and below M ~ d there is nothing to dodge.
#
# EVERY BASELINE IN THE SAME TABLE
#   Reviewers will ask for each of these by name, and a table missing one of them
#   is a table that gets rejected:
#     full softmax at width d ............... the arm to beat
#     learned dense W at width M ............ matched-capacity control, the one
#                                             that isolates "frozen and binary"
#                                             from "narrow"
#     frozen random binary Phi at width M ... isolates STRUCTURE from BINARINESS
#     Huffman hierarchical softmax .......... the classical log-V tree head
#     Oda-style independent bits ............ order 1, the prior art this extends
#     VQ-Logits-style codebook scatter ...... the one-hot corner of this family
#   The learned-W and random-binary controls run at the SAME M as the best
#   monomial arm. Run at a different width they are not controls, they are
#   different experiments.
#
# ON VOCABULARY SIZE (section 8 of the plan)
#   V=32768 is the right vocab to BUILD on and the wrong one to PUBLISH on.
#   At B=15 the minimal code is a bijection onto {0,1}^15, so the minimum Hamming
#   distance is 1, there is no error-correction slack, and the ECC-versus-semantic
#   comparison is UNDEFINED. The tail is also too thin: with a 32k BPE tokenizer
#   on this corpus even the rarest tokens occur thousands of times in 600M tokens,
#   so the data-scarcity regime the hypothesis lives in does not exist and the
#   decile crossover will not fire. Reading that as a dead hypothesis would be a
#   false negative. The (vocab) group repeats the ladder at V=131072 for exactly
#   this reason. Report BITS PER BYTE, never token perplexity, when comparing
#   across tokenizers.
#
# INSTRUMENTATION IS NOT OPTIONAL
#   Every arm carries --sch-holdout-tokens 2000 and --sch-decile-metrics 1.
#   Zero-shot vocabulary extension is the headline capability claim and the
#   frequency-decile crossover is the money plot; section 6 says instrument both
#   from day one rather than retrofitting them after the grid has run.
#
#   bash scripts/c01_sch_phase1_ladder.sh                     # depth 8, everything
#   bash scripts/c01_sch_phase1_ladder.sh --group redundancy 8  # the go/no-go alone
#   bash scripts/c01_sch_phase1_ladder.sh --seeds 1 8         # fast first pass
#   bash scripts/c01_sch_phase1_ladder.sh --group vocab 8     # the V=131k slice
#   SEMANTIC_CODES=out/codes/sem_b24.pt bash scripts/c01_sch_phase1_ladder.sh --group codes 8
#
# COST NOTE. The full grid is roughly 45 runs per seed. At three seeds that is the
# bulk of the project's compute. Prune with --group once the shape is clear; the
# plan explicitly expects the grid to be pruned rather than run exhaustively.
# ============================================================================
set -o pipefail

FORCE=0
RUN_GROUPS="redundancy ladder mlp baselines codes vocab"
SEEDS=3
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
OUT_BASE="${OUT_BASE:-out/c01_sch_ladder}"
HOLDOUT="${HOLDOUT:-2000}"
# Set to a .pt produced by scripts/code_assign.py to enable the semantic arm.
SEMANTIC_CODES="${SEMANTIC_CODES:-}"
# 131k tokenizer for the (vocab) group. Train it with
#   python -m scripts.tok_train --vocab-size 131072 --tokenizer-dir tokenizer_131k
TOKENIZER_DIR_131K="${TOKENIZER_DIR_131K:-tokenizer_131k}"
mkdir -p "$OUT_BASE"

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
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c01_d${DEPTH}.log}"
STATE="${OUT_BASE}/c01_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

# Matched token budget across every arm. base_train sizes the budget from head
# parameters, and a code head has up to 17x fewer of them, so the default would
# quietly give the code arms less data than the dense control. Pin the DENSE
# arm's Chinchilla budget everywhere.
TARGET_TOKENS="${TARGET_TOKENS:-$(python3 -m scripts.code_head_budget --depth "$DEPTH" --ratio "${RATIO:-10.5}" --tokenizer-dir "${TOKENIZER_DIR:-tokenizer}")}"

COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len ${SEQ_LEN:-2048} --target-tokens $TARGET_TOKENS \
  --target-param-data-ratio -1 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 \
  --save-every 200 --eval-every -1 \
  --sch-holdout-tokens $HOLDOUT --sch-holdout-seed 7 --sch-holdout-mode target \
  --sch-decile-metrics 1 --sch-rank-probe ${RANK_CONTEXTS:-8192} \
  --sch-eval-steps ${EVAL_STEPS:-100}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

CODE="--models base --use-code-head 1"

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
echo "  C01 Phase 1: ladder, redundancy, baselines, code assignment"
echo "  depth ${DEPTH}   d=${MODEL_DIM}   seeds ${SEEDS}"
echo "  target tokens ${TARGET_TOKENS}   holdout ${HOLDOUT} ids"
echo "  groups: ${RUN_GROUPS}"
echo "  out ${OUT_BASE}"
echo "============================================================"

# ---------------------------------------------------------------- redundancy
# THE GO/NO-GO for the headline claim. Two ways to reach a comparable width:
# raise the order at the minimal code, or raise B and stay at order 2. If the
# cheap one wins, the paper has a practical method. Both arms use an MLP g so
# that neither is silently capped at d.
if has redundancy; then
    echo ""
    echo "### REDUNDANCY vs DEPTH: B=64 k=2 (M=2080) against B=15 k=4 (M=1940)"
    run RED_B64_k2 "$DEPTH" $CODE --sch-bits 64 --sch-order 2 \
        --sch-code-mode random --sch-g-type mlp --sch-g-hidden "$MODEL_DIM"
    run RED_B15_k4 "$DEPTH" $CODE --sch-bits 15 --sch-order 4 \
        --sch-code-mode binary --sch-g-type mlp --sch-g-hidden "$MODEL_DIM"
    # The same pair with a linear g, to show the cap is what makes them differ
    # when it is present rather than an artefact of the MLP.
    run RED_B64_k2_lin "$DEPTH" $CODE --sch-bits 64 --sch-order 2 --sch-code-mode random
    run RED_B15_k4_lin "$DEPTH" $CODE --sch-bits 15 --sch-order 4 --sch-code-mode binary
fi

# ---------------------------------------------------------------- ladder
# The saturation sweep proper. B x k with a linear g. Expect quality to track M
# until M crosses the data's intrinsic rank requirement, then flatten, and expect
# the flattening point to MOVE with B. That movement is contribution 2; a single
# saturation point at one B is not a finding.
if has ladder; then
    echo ""; echo "### LADDER: B in {15, 24, 32, 64} x order in {1, 2, 3}, linear g"
    for B in 15 24 32 64; do
        MODE=binary; [ "$B" -gt 15 ] && MODE=random
        for K in 1 2 3; do
            # B=64 at order 3 is 43808 columns, above V and above the width cap.
            if [ "$B" -eq 64 ] && [ "$K" -ge 3 ]; then continue; fi
            run "LAD_B${B}_k${K}" "$DEPTH" $CODE --sch-bits "$B" --sch-order "$K" \
                --sch-code-mode "$MODE"
        done
    done
fi

# ---------------------------------------------------------------- mlp
# The slice that demonstrates section 3.4 inside the quality sweep, not just in
# the rank probe. At B=15 orders 3 and 4 are rank-identical under a linear g
# (both capped at d=512) and genuinely different under an MLP g. If quality
# tracks rank, these two arms must separate where the linear pair does not.
if has mlp; then
    echo ""; echo "### MLP g: the same rungs, uncapped"
    for K in 2 3 4; do
        run "MLP_B15_k${K}" "$DEPTH" $CODE --sch-bits 15 --sch-order "$K" \
            --sch-g-type mlp --sch-g-hidden "$MODEL_DIM"
    done
    run MLP_B32_k2 "$DEPTH" $CODE --sch-bits 32 --sch-order 2 --sch-code-mode random \
        --sch-g-type mlp --sch-g-hidden "$MODEL_DIM"
fi

# ---------------------------------------------------------------- baselines
# All of them, at matched M where matching is what makes them controls.
if has baselines; then
    echo ""; echo "### BASELINES: every arm a reviewer will ask for, same table"
    # The arm to beat. No code head at all.
    run BASE_dense_softmax "$DEPTH" --models base

    # Matched-capacity control: identical width, identical g, LEARNED real-valued
    # output embedding. This is the one that answers "is the structure doing
    # anything, or is a narrow head just fine".
    run BASE_learned_W_M1940 "$DEPTH" $CODE --sch-phi-mode learned --sch-max-m 1940
    run BASE_learned_W_M575  "$DEPTH" $CODE --sch-phi-mode learned --sch-max-m 575

    # Structure control: frozen, binary, same width, no interaction structure.
    # Density is matched to the monomial arm automatically.
    run BASE_random_binary_M1940 "$DEPTH" $CODE --sch-phi-mode random_binary --sch-max-m 1940
    run BASE_random_binary_M575  "$DEPTH" $CODE --sch-phi-mode random_binary --sch-max-m 575

    # The classical tree head. Structurally different from an interaction
    # expansion, and the standard answer to "why not hierarchical softmax".
    run BASE_hsoftmax "$DEPTH" --models base --use-code-head 1 --sch-head-type hsoftmax

    # Oda et al. 2017: independent bits at the minimal code. The prior art this
    # work extends, and the rank-15 arm the theory says should be bad. Trained
    # with exact cross-entropy rather than their per-bit BCE, which makes this a
    # STRONGER version of their method: same rank bound, exactly normalised.
    run BASE_oda_order1 "$DEPTH" $CODE --sch-order 1 --sch-code-mode binary

    # VQ-Logits: a K-vector codebook scattered to the vocabulary, which is
    # exactly the one-hot corner of this family. Its per-token bias is what
    # recovers most of the lost quality, so it is paired with --sch-bias 1.
    run BASE_vqlogits_K512 "$DEPTH" $CODE --sch-phi-mode onehot --sch-max-m 512 --sch-bias 1
fi

# ---------------------------------------------------------------- codes
# The code assignment axis, at a fixed (B, k). The semantic and ECC objectives
# are directly opposed: an ECC maximises minimum Hamming distance, generalisation
# wants semantic neighbours at SMALL Hamming distance. --sch-code-ecc-bits sweeps
# the tension on one axis by adding parity bits to a semantic base code.
if has codes; then
    echo ""; echo "### CODES: assignment arms at B=24, order 3 (M=2324)"
    run CODE_random    "$DEPTH" $CODE --sch-bits 24 --sch-order 3 --sch-code-mode random
    run CODE_ecc       "$DEPTH" $CODE --sch-bits 24 --sch-order 3 --sch-code-mode ecc
    run CODE_frequency "$DEPTH" $CODE --sch-bits 24 --sch-order 3 --sch-code-mode frequency

    if [ -n "$SEMANTIC_CODES" ] && [ -f "$SEMANTIC_CODES" ]; then
        run CODE_semantic "$DEPTH" $CODE --sch-order 3 \
            --sch-code-mode file --sch-code-path "$SEMANTIC_CODES"
        # The tension sweep. Parity bits raise minimum distance without
        # disturbing the semantic base assignment, so this walks from the
        # generalisation end of the axis to the error-correction end.
        for P in 4 8 16; do
            run "CODE_semantic_ecc${P}" "$DEPTH" $CODE --sch-order 3 \
                --sch-code-mode file --sch-code-path "$SEMANTIC_CODES" \
                --sch-code-ecc-bits "$P"
        done
    else
        echo "SKIP  semantic code arms: no code matrix at '${SEMANTIC_CODES:-<unset>}'."
        echo "      Build one first, from a trained dense checkpoint:"
        echo "        python -m scripts.code_assign --mode semantic --semantic-method itq \\"
        echo "          --bits 24 --from-checkpoint <ckpt-dir> --out out/codes/sem_b24.pt --report"
        echo "      then re-run with SEMANTIC_CODES=out/codes/sem_b24.pt"
    fi
fi

# ---------------------------------------------------------------- vocab
# The axis the thesis lives on. One point is not a trend, and 32768 is the wrong
# point to publish. At 131k the rarest deciles drop to tens or hundreds of
# occurrences, which is the data-scarcity regime where codes are supposed to win.
if has vocab; then
    echo ""; echo "### VOCAB: the same rungs at V=131072 (bits per byte, not perplexity)"
    # Builds the tokenizer, its token_bytes and its frequency table if any are
    # missing, and is a no-op otherwise. Section 8 requires a separate tokenizer
    # per vocabulary size on the same corpus; ensure_tokenizer also refuses a
    # directory whose vocab_size is not 131072, so these arms cannot silently
    # run at the wrong vocabulary.
    if ! python3 -m scripts.ensure_tokenizer --vocab-size 131072 \
            --tokenizer-dir "${TOKENIZER_DIR_131K}" \
            --data-dir "${DATA_DIR:-data}" ${MAX_SHARDS:+--max-shards "$MAX_SHARDS"}; then
        echo "SKIP  V=131072 arms: could not prepare '${TOKENIZER_DIR_131K}'."
    else
        V131="--tokenizer-dir ${TOKENIZER_DIR_131K}"
        run V131_dense    "$DEPTH" --models base $V131
        run V131_B17_k2   "$DEPTH" $CODE $V131 --sch-order 2
        run V131_B17_k3   "$DEPTH" $CODE $V131 --sch-order 3
        run V131_B64_k2   "$DEPTH" $CODE $V131 --sch-bits 64 --sch-order 2 \
            --sch-code-mode random --sch-g-type mlp --sch-g-hidden "$MODEL_DIM"
    fi
fi

echo ""
echo "============================================================"
echo "  C01 depth ${DEPTH} complete."
echo ""
echo "  READ: sch_results.csv. The columns that decide things:"
echo "    val_bpb                  quality, comparable across arms at fixed V"
echo "    phi_width_M              the width the ladder is indexed by"
echo "    rank_effective_rank      achieved rank, against the ~1000 threshold"
echo "    bpb_decile0..9           the money plot. Looking for a CROSSOVER:"
echo "                             the code head losing on head tokens and"
echo "                             winning on tail tokens versus BASE_dense_softmax."
echo "    bpb_tail_minus_head      the crossover as one number"
echo "    bpb_holdout / holdout_*  zero-shot vocabulary extension"
echo ""
echo "  DECISIONS THIS SWEEP FEEDS (section 7 kill criteria):"
echo "    * order 3 or B=64 still more than 10% worse than softmax at V=131k:"
echo "      codes underperform theory. Diagnose before scaling."
echo "    * no frequency-decile crossover in ANY configuration:"
echo "      drop the tail-generalisation framing, pivot to extension-only."
echo "    * RED_B64_k2 at least matching RED_B15_k4:"
echo "      redundancy beats depth. That is the quotable result; carry that"
echo "      configuration into c02 and c03."
echo "============================================================"

done
