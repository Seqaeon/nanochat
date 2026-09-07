#!/usr/bin/env bash
# ============================================================================
# EET P02 — the three decision tests
# ============================================================================
# EET currently trades ~0.06 val_bpb against a ~25% wallclock speedup at d8. On the
# repo's iso-data dense curve (mst_isodata.html, d10..d16, local log-log slope about
# -0.085 near d8) that speedup is only worth ~0.025 bpb, and the ~39% FLOP cut the
# bell schedule actually makes is worth ~0.042 bpb. So EET is Pareto-dominated and
# needs to give back roughly 0.02 (FLOPs) to 0.035 (wallclock) of the 0.06.
#
# eet_experiment_log.md concludes the remainder is "architectural". Thirteen ideas were
# tried and abandoned; every one of them addressed gradient flow, learning rates,
# representation alignment, distillation or scheduling. None addressed the two things
# that actually differ from dense:
#
#   Defect 1  CONTEXT DESTRUCTION. An exited token loses its keys and values in every
#             later layer, so at d8 the survivors -- the hard tokens -- attend over
#             12% to 27% of the sequence. Restoring full-context reads costs about
#             1.6% of a dense layer's FLOPs.
#   Defect 2  DATA STARVATION. use_pos_embed is off, so the global router reads only
#             norm(wte(idx)): exit depth is a per-vocabulary-item lookup table. Layer 7
#             trains on a fixed ~10% slice of the vocabulary for the entire run and
#             never sees "the". That explains the measured "78% of the gap is backbone
#             co-training" and why depth-LR-scale and depth-grad-scale did nothing:
#             a deep layer short of SAMPLES cannot be fixed with a learning rate.
#
# This sweep decides, with pre-registered criteria, whether either defect is real.
# It is a GO/NO-GO on the direction, not an attempt to ship a paper number.
#
# PRE-REGISTERED CRITERIA (do not edit after seeing results)
#   T0A gate   delta_bpb(freq/ctx) on the dense checkpoint >= 0.020, else Defect 1 is
#              closed and the T1 arms are skipped automatically.
#   T0B gate   token-mass reaching the last layer <= 0.50 under deterministic routing,
#              else Defect 2 is closed and the T2 arms are skipped automatically.
#   T1 pass    best kv-mode arm reaches gap <= 0.045 bpb vs DENSE.
#   T2 pass    best route-noise arm reaches gap <= 0.045 bpb vs DENSE.
#   T3 pass    combined arm reaches gap <= break_even_wallclock (about 0.025 at d8),
#              at two depths, with the gap not widening from d8 to d16.
#   If T1 and T2 both fail, the "0.06 is architectural" verdict is confirmed. Close the
#   direction rather than sweeping more flags.
#
# USAGE
#   bash scripts/eet_p02_tests.sh [--force] [--skip-gates] [DEPTH]
#   DEPTH=8 by default. Run d8 first, then d16 only if T3 passed at d8.
#
# ENV
#   EET_OUT_BASE    output root                 (default out/eet_p02)
#   TARGET_FRAC     eet_target_active_frac      (default 0.10, matches the P01 runs)
#   DEVICE_BATCH_SIZE / DATA_DIR / TOKENIZER_DIR / MAX_SHARDS / NPROC_PER_NODE
#   ORACLE_EVAL_TOKENS  tokens for the T0A oracle (default 2000000)
# ============================================================================

set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-.}"

FORCE=0
SKIP_GATES=0
REDO_ORACLE=0
REDO_TAGS=""
DEPTH=8
while [ $# -gt 0 ]; do
    case $1 in
        --force)       FORCE=1 ;;
        --skip-gates)  SKIP_GATES=1 ;;
        --redo-oracle) REDO_ORACLE=1 ;;
        --redo)        shift; REDO_TAGS="$REDO_TAGS $1" ;;
        --help|-h)
            sed -n '2,52p' "$0"; exit 0 ;;
        *)             DEPTH=$1 ;;
    esac
    shift
done

EET_OUT_BASE="${EET_OUT_BASE:-out/eet_p02}"
LOGFILE="${SWEEP_LOG:-${EET_OUT_BASE}/sweep_eet_p02_d${DEPTH}.log}"
STATE_FILE="${EET_OUT_BASE}/sweep_state_d${DEPTH}.json"
TARGET_FRAC="${TARGET_FRAC:-0.10}"
ORACLE_EVAL_TOKENS="${ORACLE_EVAL_TOKENS:-2000000}"
mkdir -p "$EET_OUT_BASE"

echo "==============================================================="
echo "  EET P02: context restoration and stochastic routing"
echo "  Depth:            ${DEPTH}"
echo "  Target active:    ${TARGET_FRAC}"
echo "  Output:           ${EET_OUT_BASE}"
echo "  Log:              ${LOGFILE}"
echo "==============================================================="

# ---------------------------------------------------------------- state ----
init_state() { [ -f "$STATE_FILE" ] || echo '{"completed":{},"started":{},"vars":{}}' > "$STATE_FILE"; }

check_completed() {
    [ "$FORCE" -eq 1 ] && return 1
    python3 -c "
import json, sys
s = json.load(open('$STATE_FILE'))
sys.exit(0 if '$1' in s.get('completed', {}) else 1)" 2>/dev/null
}

mark_started() {
    python3 -c "
import json, datetime
s = json.load(open('$STATE_FILE'))
s.setdefault('started', {})['$1'] = {'run_dir': '$2', 'started_at': datetime.datetime.now().isoformat()}
json.dump(s, open('$STATE_FILE','w'), indent=2)"
}

mark_completed() {
    python3 -c "
import json, datetime
s = json.load(open('$STATE_FILE'))
s.setdefault('completed', {})['$1'] = {'completed_at': datetime.datetime.now().isoformat()}
s.get('started', {}).pop('$1', None)
json.dump(s, open('$STATE_FILE','w'), indent=2)"
}

set_var() {
    python3 -c "
import json
s = json.load(open('$STATE_FILE'))
s.setdefault('vars', {})['$1'] = '$2'
json.dump(s, open('$STATE_FILE','w'), indent=2)"
}

get_var() {
    python3 -c "
import json
print(json.load(open('$STATE_FILE')).get('vars', {}).get('$1', ''))" 2>/dev/null
}

print_header() {
    echo ""
    echo "---------------------------------------------------------------"
    echo "  [$1]  $2"
    echo "---------------------------------------------------------------"
}

init_state

for _tag in $REDO_TAGS; do
    python3 -c "
import json
s = json.load(open('$STATE_FILE'))
gone = s.get('completed', {}).pop('$_tag', None) is not None
s.get('started', {}).pop('$_tag', None)
json.dump(s, open('$STATE_FILE','w'), indent=2)
print('[redo] cleared $_tag' if gone else '[redo] $_tag was not marked completed')"
done

# ---------------------------------------------------- tokenizer preflight ---
# The repo ships a 265-token stub at ./tokenizer (tokenizer.pkl is 1.9 KB). base_train
# prints "Vocab size: 265" and trains happily, so a whole sweep can complete against a
# byte-level vocabulary and produce bpb numbers that mean nothing and cannot be compared
# with anything trained at V=32768. Refuse to start instead.
TOK_PROBE=$(PYTHONPATH=. python -c "
from nanochat.tokenizer import get_tokenizer
print(get_tokenizer('${TOKENIZER_DIR:-tokenizer}').get_vocab_size())" 2>&1) || TOK_PROBE=""
TOK_VOCAB=$(printf '%s' "$TOK_PROBE" | tail -1 | tr -dc '0-9')
if [ -z "$TOK_VOCAB" ]; then
    echo ""
    echo "[ABORT] could not read a vocabulary from '${TOKENIZER_DIR:-tokenizer}':"
    printf '%s\n' "$TOK_PROBE" | tail -3 | sed 's/^/        /'
    exit 1
fi
if [ "$TOK_VOCAB" -lt 1000 ]; then
    echo ""
    echo "[ABORT] --tokenizer-dir '${TOKENIZER_DIR:-tokenizer}' resolves to vocab_size=${TOK_VOCAB}."
    echo "        That is the byte-level stub, not a trained tokenizer. Every run would"
    echo "        train at that vocabulary and every bpb would be meaningless."
    echo "        Point TOKENIZER_DIR at a real one and rerun any arm already trained"
    echo "        against the stub with --redo <TAG>."
    exit 1
fi
echo "  Tokenizer:        ${TOKENIZER_DIR:-tokenizer} (vocab_size ${TOK_VOCAB})"

# ------------------------------------------------------------- common ------
# Every arm after DENSE is pinned to DENSE's exact token count via --target-tokens,
# so the whole sweep is iso-data. Without that pin the EET arms would get a slightly
# different Chinchilla budget from the router's extra parameters, which is the same
# order as the effect being measured.
EET_COMMON="--models base \
  --device-batch-size ${DEVICE_BATCH_SIZE:-128} --total-batch-size -1 \
  --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-tokens -1 --target-active-params 0 \
  --save-every 200 --eval-every -1"
[ -n "${MAX_SHARDS:-}" ]  && EET_COMMON="$EET_COMMON --max-shards $MAX_SHARDS"
[ "${USE_FP8:-0}" = "1" ] && EET_COMMON="$EET_COMMON --fp8"

# The P01 configuration that produced the ~0.06 gap. Held fixed across every arm so the
# only thing that varies is the mechanism under test.
EET_BASE_FLAGS="--use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
  --eet-router-type mlp1 \
  --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
  --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
  --eet-capacity-schedule bell \
  --eet-global-router 1 \
  --eet-min-exit-layer 1 \
  --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
  --eet-compute-skip 1 --eet-target-active-frac ${TARGET_FRAC} \
  --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
  --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 \
  --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0 \
  --eet-capacity-alignment-lambda 0.0"

run_dir_for() { echo "${EET_OUT_BASE}/$1/depth_${2:-$DEPTH}"; }

run_experiment() {
    local tag="$1"; shift
    local desc="$1"; shift
    local depth="${RUN_DEPTH:-$DEPTH}"

    if check_completed "$tag"; then
        echo "[skip] $tag already completed"
        return 0
    fi
    print_header "$tag" "$desc"
    local run_dir="${EET_OUT_BASE}/${tag}"
    if [ "$FORCE" -eq 1 ] && [ -d "$run_dir" ]; then
        echo "[force] removing $run_dir"
        rm -rf "$run_dir"
    fi
    mark_started "$tag" "$run_dir"
    if bash scripts/research_sweep.sh $EET_COMMON --out-dir "$run_dir" "$@" "$depth" 2>&1 | tee -a "$LOGFILE"; then
        echo "[ok] $tag"
        mark_completed "$tag"
    else
        echo "[FAIL] $tag -- will retry on the next invocation"
        return 1
    fi
}

# ============================================================================
# P02b -- ONE ARM ONLY.
#
# Everything from the original P02 is commented out below. It was all measured with
# --eet-router-task-grad 1, which breaks torch.compile with
#     scatter(): Expected self.dtype to be equal to src.dtype
# and falls back to eager, so every wallclock number in that sweep compared compiled
# dense against uncompiled EET. Measured here, interleaved in one process:
#
#     dense                192.0 ms   1.000x
#     eet, task-grad ON    254.5 ms   1.326x     <- what the sweep ran
#     eet, task-grad OFF   134.7 ms   0.702x     <- against a 0.722x analytic floor
#     eet, uncompiled      230.6 ms   1.201x     <- the fallback the sweep actually got
#
# This arm is the 0.702x configuration. It is the only change from EET_BASE_D8.
# ============================================================================

DENSE_PIN=""
_stored="$(get_var tokens_d${DEPTH})"
if [ -n "$_stored" ] && [ "$(get_var tokens_vocab_d${DEPTH})" = "$TOK_VOCAB" ]; then
    DENSE_PIN="--target-tokens $_stored"
    echo "[iso-data] DENSE pinned to the stored budget ${_stored}"
fi
run_experiment "DENSE_D${DEPTH}" \
    "Dense control (no early exit). Reference for the gap." \
    --use-eet 0 $DENSE_PIN

TOKENS="$(get_var tokens_d${DEPTH})"
ISO_DATA=""
[ -n "$TOKENS" ] && ISO_DATA="--target-tokens $TOKENS"

# --eet-router-task-grad 0 alone is NOT enough. Measured per-flag on one GPU, one process:
#     bare        157.5 ms
#     +gumbel     160.0   (+2.5)
#     +ce_guided  185.0   (+27.5)
#     +cap_align  189.5   (+32.0)
#     +ema        203.8   (+46.3)
# The three auxiliary terms cost ~106 ms between them, about as much as task-grad did.
# This arm removes all of it. The router then receives no gradient and is a frozen
# projection, which is the honest control: random routing was measured to tie the learned
# router (1.06487 vs 1.06433), so nothing of value is being switched off.
run_experiment "EET_FAST_D${DEPTH}" \
    "EET routing with no aux loss and no task-grad: the 0.702x configuration" \
    $EET_BASE_FLAGS $ISO_DATA \
    --eet-router-task-grad 0 \
    --eet-loss-variant none \
    --eet-depth-weight-type none \
    --eet-capacity-alignment-lambda 0.0 \
    --eet-surprise-lambda 0.0 \
    --eet-ce-guided-lambda 0.0 || true

# ---- ORIGINAL P02 ARMS, DISABLED -------------------------------------------
# ============================================================================
# Step 1 -- DENSE control. Needed three times over: as the gap reference, as the
#           subject of the T0A oracle, and as the source of the iso-data token count.
# ============================================================================
#DENSE_PIN=""
#_stored="$(get_var tokens_d${DEPTH})"
#if [ -n "$_stored" ] && [ "$(get_var tokens_vocab_d${DEPTH})" = "$TOK_VOCAB" ]; then
#    DENSE_PIN="--target-tokens $_stored"
#    echo "[iso-data] DENSE pinned to the stored budget ${_stored}"
#fi
#run_experiment "DENSE_D${DEPTH}" \
#    "Dense control (no early exit). Reference for every gap in this sweep." \
#    --use-eet 0 $DENSE_PIN --target-tokens 265814016

# Pin every later arm to this exact token budget.
# The stored budget is only valid for the vocabulary it was measured at: a d8 model at
# V=32768 gets 440,401,920 Chinchilla tokens, at V=265 it gets 265,814,016. Reusing the
# stale one silently trained DENSE_D8 on 1.66x the data of every arm it was the control
# for, which is not an iso-data comparison at all.
#TOKENS="$(get_var tokens_d${DEPTH})"
#TOKENS_VOCAB="$(get_var tokens_vocab_d${DEPTH})"
#if [ -n "$TOKENS" ] && [ "$TOKENS_VOCAB" != "$TOK_VOCAB" ]; then
#    echo "[iso-data] stored budget ${TOKENS} was measured at vocab ${TOKENS_VOCAB:-unknown},"
#    echo "           but the tokenizer is now ${TOK_VOCAB}. Discarding it and remeasuring."
#    TOKENS=""
#fi
#if [ -z "$TOKENS" ]; then
#    # head -1, not tail -1: DENSE is always the first run in the log, and on a resumed
#    # sweep the last occurrence would belong to whichever arm ran most recently.
#    TOKENS=""
#    if [ -f "$LOGFILE" ]; then
#        TOKENS=$(grep -h "Total number of training tokens:" "$LOGFILE" 2>/dev/null \
#                 | head -1 | sed 's/.*: *//' | tr -d ', ' || true)
#    fi
#    if [ -n "$TOKENS" ]; then
#        set_var "tokens_d${DEPTH}" "$TOKENS"
#        set_var "tokens_vocab_d${DEPTH}" "$TOK_VOCAB"
#        echo "[iso-data] pinning all later arms to ${TOKENS} tokens (vocab ${TOK_VOCAB})"
#    else
#        echo "[warn] no dense token count in $LOGFILE and none stored in the state file."
#        echo "       Later arms will use their own Chinchilla budget, so the sweep is NOT"
#        echo "       iso-data. Set it by hand:"
#        echo "         python3 -c \"import json;s=json.load(open('$STATE_FILE'));s.setdefault('vars',{})['tokens_d${DEPTH}']='<N>';json.dump(s,open('$STATE_FILE','w'),indent=2)\""
#    fi
#fi
#ISO_DATA=""
#[ -n "$TOKENS" ] && ISO_DATA="--target-tokens $TOKENS"

# ============================================================================
# Step 2 -- T0A: the context oracle. Runs on the dense checkpoint in minutes and
#           decides whether the T1 training arms are worth any GPU time at all.
# ============================================================================
#DENSE_CKPT="$(run_dir_for "DENSE_D${DEPTH}")/ckpt_base/base"
#ORACLE_JSON="${EET_OUT_BASE}/oracle_d${DEPTH}.json"

#[ "$REDO_ORACLE" -eq 1 ] && rm -f "$ORACLE_JSON"
#echo "  Oracle result:    $ORACLE_JSON"
#if [ -f "$ORACLE_JSON" ]; then
#    echo "  Oracle:           present, will NOT rerun (delete that exact file, or pass --redo-oracle)"
#elif [ ! -d "$DENSE_CKPT" ]; then
#    echo "  Oracle:           SKIPPED, no dense checkpoint at $DENSE_CKPT"
#else
#    echo "  Oracle:           will run"
#fi

#if [ ! -f "$ORACLE_JSON" ] && [ -d "$DENSE_CKPT" ]; then
#    print_header "T0A" "Context oracle on the dense checkpoint (no training)"
#    FREQ_N=$(PYTHONPATH=. python -c "
#import torch; print(torch.load('${TOKENIZER_DIR:-tokenizer}/freq_table.pt', weights_only=True).numel())" 2>/dev/null | tr -dc '0-9')
#    if [ "${FREQ_N:-0}" != "$TOK_VOCAB" ]; then
#        echo "[ABORT] ${TOKENIZER_DIR:-tokenizer}/freq_table.pt has ${FREQ_N} entries but the"
#        echo "        tokenizer has ${TOK_VOCAB}. The oracle's 'freq' ranking would be wrong."
#        echo "        Rebuild it: python -m scripts.code_assign --build-freq-table"
#        exit 1
#    fi
#    python -m scripts.eet_context_oracle \
#        --ckpt-dir "$DENSE_CKPT" \
#        --data-dir "${DATA_DIR:-data}" --tokenizer-dir "${TOKENIZER_DIR:-tokenizer}" \
#        ${MAX_SHARDS:+--max-shards $MAX_SHARDS} \
#        --target-active-frac "$TARGET_FRAC" --min-exit-layer 1 \
#        --capacity-schedule bell \
#        --rank freq,random,ce --ablation ctx,depth,both \
#        --eval-tokens "$ORACLE_EVAL_TOKENS" \
#        --gate-delta-bpb 0.02 \
#        --out "$ORACLE_JSON" 2>&1 | tee -a "$LOGFILE" || true
#fi

#if [ ! -f "$ORACLE_JSON" ]; then
#    echo ""
#    echo "[GATE T0A] The oracle produced no result. Failing OPEN: the T1 arms will run"
#    echo "           anyway, so the sweep still answers the question, but the cheap"
#    echo "           screen was lost. Fix the oracle and rerun it to save the GPU time."
#fi

#T1_ENABLED=1
#if [ "$SKIP_GATES" -eq 0 ] && [ -f "$ORACLE_JSON" ]; then
#    T1_ENABLED=$(python3 -c "
#import json
#r = json.load(open('$ORACLE_JSON'))
#print(1 if r.get('gate_passed', True) else 0)")
#    if [ "$T1_ENABLED" -eq 0 ]; then
#        echo ""
#        echo "[GATE T0A FAILED] Context destruction costs less than 0.020 bpb on the dense"
#        echo "                  checkpoint. Defect 1 is closed. Skipping the T1 arms."
#        echo "                  Re-run with --skip-gates to force them anyway."
#    fi
#fi

# ============================================================================
# Step 3 -- EET baseline. Reproduces the ~0.06 gap under this sweep's exact
#           token budget, so the gap is measured rather than quoted.
# ============================================================================
#run_experiment "EET_BASE_D${DEPTH}" \
#    "EET baseline: P01 config, kv-mode none, deterministic routing. Reproduces the gap." \
#    $EET_BASE_FLAGS $ISO_DATA || true

# ============================================================================
# Step 4 -- T0B: routing coverage. A short EET run with the per-layer vocabulary
#           counter on. Decides whether the T2 arms are worth GPU time.
#           --compile 0 because the bincount breaks the graph.
# ============================================================================
#COVERAGE_TAG="T0B_COVERAGE_D${DEPTH}"
#if ! check_completed "$COVERAGE_TAG"; then
#    run_experiment "$COVERAGE_TAG" \
#        "Routing coverage diagnostic: which vocabulary reaches which layer (200 steps)" \
#        $EET_BASE_FLAGS $ISO_DATA \
#        --num-iterations 200 --eval-every 100 --no-compile \
#        --eet-coverage-diag 1 || true
#fi

#T2_ENABLED=1
#if [ "$SKIP_GATES" -eq 0 ]; then
#    # The JSON is written to a file rather than interpolated into the python -c
#    # string: it contains double quotes, which would terminate the shell argument.
#    COV_JSON="${EET_OUT_BASE}/coverage_d${DEPTH}.json"
#    if [ -f "$LOGFILE" ]; then
#        grep -h "EET_COVERAGE_JSON" "$LOGFILE" 2>/dev/null | tail -1 \
#            | sed 's/.*EET_COVERAGE_JSON //' > "$COV_JSON" || true
#    fi
#    if [ -s "$COV_JSON" ]; then
#        T2_ENABLED=$(python3 -c "
#import json, sys
#rows = json.load(open('$COV_JSON'))
#deep = rows[-1]['mass_frac']
#print('deep-layer token mass reaching the last layer: %.4f' % deep, file=sys.stderr)
#print(1 if deep <= 0.50 else 0)")
#        if [ "$T2_ENABLED" -eq 0 ]; then
#            echo ""
#            echo "[GATE T0B FAILED] The deepest layer already sees more than half the token"
#            echo "                  mass. Defect 2 is closed. Skipping the T2 arms."
#        fi
#    fi
#fi

# ============================================================================
# Step 5 -- TEST 1: context restoration.
#   fresh  keys/values re-projected at every layer for every position. Quality upper
#          bound; if this does not move the gap, no cheaper variant will either.
#   stale  keys/values banked at each token's exit layer. Near free, and the variant
#          a paper would actually ship.
# Run 'fresh' first: it is the gate on the whole family.
# ============================================================================
#if [ "$T1_ENABLED" -eq 1 ]; then
#    run_experiment "T1_FRESH_D${DEPTH}" \
#        "Test 1a: exited tokens keep fresh per-layer keys/values (quality upper bound)" \
#        $EET_BASE_FLAGS $ISO_DATA --eet-kv-mode fresh || true

#    run_experiment "T1_STALE_D${DEPTH}" \
#        "Test 1b: exited tokens keep the keys/values banked at their exit layer (near free)" \
#        $EET_BASE_FLAGS $ISO_DATA --eet-kv-mode stale || true
#fi

# ============================================================================
# Step 6 -- TEST 2: stochastic routing. Identical per-step FLOP budget; only the
#   ASSIGNMENT is resampled, so every token id visits every depth over training.
#   random  noise dominates the router: the clean scientific arm. If uniform-random
#           routing beats the learned deterministic router, the gap is coverage, not
#           routing quality, and that is the whole finding.
#   anneal  explore then commit: the arm a paper would actually ship.
# ============================================================================
#if [ "$T2_ENABLED" -eq 1 ]; then
#    run_experiment "T2_RANDOM_D${DEPTH}" \
#        "Test 2a: effectively uniform-random routing (noise 10.0), same capacities" \
#        $EET_BASE_FLAGS $ISO_DATA --eet-route-noise 10.0 || true

#    run_experiment "T2_ANNEAL_D${DEPTH}" \
#        "Test 2b: stochastic routing annealed to deterministic (1.0 -> 0.0)" \
#        $EET_BASE_FLAGS $ISO_DATA --eet-route-noise 1.0 --eet-route-noise-end 0.0 || true
#fi

# ============================================================================
# Step 7 -- TEST 3: both mechanisms, plus the iso-FLOP dense controls the Pareto
#   claim actually needs. The bell schedule at target ${TARGET_FRAC} leaves an
#   average active fraction of about 0.61 at d8, so the honest dense control is a
#   dense model at that active-FLOP budget and the SAME token count. d5 and d6
#   bracket it; the report interpolates.
# ============================================================================
#if [ "$T1_ENABLED" -eq 1 ] || [ "$T2_ENABLED" -eq 1 ]; then
#    T3_FLAGS="$EET_BASE_FLAGS $ISO_DATA"
#    [ "$T1_ENABLED" -eq 1 ] && T3_FLAGS="$T3_FLAGS --eet-kv-mode stale"
#    [ "$T2_ENABLED" -eq 1 ] && T3_FLAGS="$T3_FLAGS --eet-route-noise 1.0 --eet-route-noise-end 0.0"
#    run_experiment "T3_BOTH_D${DEPTH}" \
#        "Test 3: shippable combination of whichever mechanisms survived their gates" \
#        $T3_FLAGS || true
#fi

#for CTRL_DEPTH in 5 6; do
#    RUN_DEPTH=$CTRL_DEPTH run_experiment "DENSE_ISOFLOP_D${CTRL_DEPTH}_FOR_D${DEPTH}" \
#        "Iso-FLOP iso-data dense control at depth ${CTRL_DEPTH} (brackets EET's active-FLOP budget)" \
#        --use-eet 0 $ISO_DATA || true
#done

# ============================================================================
# Step 8 -- report against the pre-registered criteria
# ============================================================================
print_header "REPORT" "Break-even arithmetic and pass/fail"
python -m scripts.eet_p02_report \
    --out-base "$EET_OUT_BASE" --depth "$DEPTH" \
    --target-active-frac "$TARGET_FRAC" \
    --oracle "$ORACLE_JSON" \
    --log "$LOGFILE" 2>&1 | tee -a "$LOGFILE" || true
