#!/usr/bin/env bash
# ============================================================================
# EET Tier 0 — is the depth-collapse diagnosis right, and is the trade winnable?
# ============================================================================
# WHAT IS ESTABLISHED
#   EET is faster: 0.863x dense on an H200 at B=128 (fwd 0.821, bwd 0.858, opt 1.436).
#   EET's routing is worthless: 1.0649 random / 1.0643 learned / 1.0610 with the router
#   and every auxiliary loss removed. It is a fixed, content-blind depth schedule.
#   EET's backbone collapses. Per-layer bpb through the shared head:
#       dense  2.676 2.360 2.244 1.783 1.686 1.296 1.194 0.982   (gains 0.80 over L4-7)
#       EET    2.107 1.267 1.188 1.115 1.113 1.113 1.114 1.114   (gains 0.001)
#   EET's layer 3 is 0.67 bpb BETTER than dense's layer 3, and its layer 7 is 0.13 WORSE
#   than dense's layer 7. That is a trade, not starvation: forcing early layers to be
#   prediction-readable buys early quality and destroys the depth hierarchy.
#
# THE HYPOTHESIS
#   One residual stream cannot be simultaneously readable-now and refinable-later. If so,
#   removing the early-readability pressure should wake the deep layers up (T0A), and a
#   healthy backbone should pay a measurable price for being readable everywhere (T0B).
#
# PRE-REGISTERED KILL CRITERIA -- do not edit after seeing results
#   T0A  min_exit_layer 1 -> 4, so layers 1-4 are pure feature builders.
#        PASS if the last-4-layer bpb gain rises above 0.05 (it is 0.001 today) AND
#        mean |dx| at layers 5-7 rises above 5 (it is 1.3-1.9 today).
#        FAIL  -> premature readability is NOT the mechanism. Tiers 1 and 4 of the
#                brainstorm are dead and I am wrong about the cause. Say so.
#
#   T0B  dense + deep supervision at every layer through the shared head.
#        PASS if dense loses less than 0.03 bpb against plain dense, i.e. a healthy
#        backbone CAN be readable at every depth cheaply, so the trade is winnable.
#        FAIL  -> readability at every depth costs more than the whole EET gap, so no
#                architecture can have both and the direction closes on a bound rather
#                than on a failed experiment.
#
#   Both are diagnostics, not contributions. Neither is expected to be Pareto-competitive:
#   T0A gives up most of the FLOP saving and T0B is a dense model.
#
# USAGE
#   TOKENIZER_DIR=<real> bash scripts/eet_tier0.sh [--force] [--redo TAG] [DEPTH]
# ============================================================================

set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-.}"

FORCE=0; REDO_TAGS=""; DEPTH=8
while [ $# -gt 0 ]; do
    case $1 in
        --force) FORCE=1 ;;
        --redo)  shift; REDO_TAGS="$REDO_TAGS $1" ;;
        --help|-h) sed -n '2,45p' "$0"; exit 0 ;;
        *)       DEPTH=$1 ;;
    esac
    shift
done

OUT_BASE="${OUT_BASE:-out/eet_tier0}"
LOGFILE="${SWEEP_LOG:-${OUT_BASE}/tier0_d${DEPTH}.log}"
STATE_FILE="${OUT_BASE}/state_d${DEPTH}.json"
TARGET_FRAC="${TARGET_FRAC:-0.10}"
DS_LAMBDA="${DS_LAMBDA:-1.0}"
mkdir -p "$OUT_BASE"

echo "==============================================================="
echo "  EET Tier 0   depth=${DEPTH}   target_active=${TARGET_FRAC}"
echo "  deep-supervision lambda: ${DS_LAMBDA}"
echo "  Log: ${LOGFILE}"
echo "==============================================================="

[ -f "$STATE_FILE" ] || echo '{"completed":{},"vars":{}}' > "$STATE_FILE"
_py() { python3 -c "$1"; }
check_completed() { [ "$FORCE" -eq 1 ] && return 1; _py "
import json,sys; s=json.load(open('$STATE_FILE'))
sys.exit(0 if '$1' in s.get('completed',{}) else 1)" 2>/dev/null; }
mark_completed() { _py "
import json,datetime; s=json.load(open('$STATE_FILE'))
s.setdefault('completed',{})['$1']={'at':datetime.datetime.now().isoformat()}
json.dump(s,open('$STATE_FILE','w'),indent=2)"; }
get_var() { _py "
import json; print(json.load(open('$STATE_FILE')).get('vars',{}).get('$1',''))" 2>/dev/null; }
set_var() { _py "
import json; s=json.load(open('$STATE_FILE'))
s.setdefault('vars',{})['$1']='$2'; json.dump(s,open('$STATE_FILE','w'),indent=2)"; }

for t in $REDO_TAGS; do
    _py "
import json; s=json.load(open('$STATE_FILE'))
g=s.get('completed',{}).pop('$t',None) is not None
json.dump(s,open('$STATE_FILE','w'),indent=2)
print('[redo] cleared $t' if g else '[redo] $t was not completed')"
done

# --- tokenizer preflight: the repo has shipped a 265-token stub before ------
TOK_PROBE=$(PYTHONPATH=. python -c "
from nanochat.tokenizer import get_tokenizer
print(get_tokenizer('${TOKENIZER_DIR:-tokenizer}').get_vocab_size())" 2>&1) || TOK_PROBE=""
TOK_VOCAB=$(printf '%s' "$TOK_PROBE" | tail -1 | tr -dc '0-9')
if [ -z "$TOK_VOCAB" ] || [ "$TOK_VOCAB" -lt 1000 ]; then
    echo "[ABORT] '${TOKENIZER_DIR:-tokenizer}' gives vocab_size='${TOK_VOCAB:-unreadable}'."
    printf '%s\n' "$TOK_PROBE" | tail -2 | sed 's/^/        /'
    exit 1
fi
echo "  Tokenizer: ${TOKENIZER_DIR:-tokenizer} (vocab ${TOK_VOCAB})"

COMMON="--models base \
  --device-batch-size ${DEVICE_BATCH_SIZE:-128} --total-batch-size -1 \
  --use-onecycle 0 --log-every 200 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 --save-every 200 --eval-every -1"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

# The EET configuration that is actually worth studying: no router gradient, no auxiliary
# losses. It scores 1.0610 against the full apparatus's 1.0643 and compiles, so nothing of
# value is switched off and the timings are real rather than an eager fallback.
EET_FLAGS="--use-eet 1 --eet-compute-skip 1 --eet-global-router 1 \
  --eet-capacity-schedule bell --eet-target-active-frac ${TARGET_FRAC} \
  --eet-warmup-frac 0.0 --eet-explore-frac 0.0 --eet-router-type mlp1 \
  --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
  --eet-router-task-grad 0 --eet-loss-variant none \
  --eet-depth-weight-type none --eet-capacity-alignment-lambda 0.0 \
  --eet-surprise-lambda 0.0 --eet-ce-guided-lambda 0.0 \
  --eet-frozen-kv 0 --eet-reenter-final 0 --eet-exit-adapter-rank 0 \
  --eet-router-after-block 0 --eet-ffn-skip 0"

run_experiment() {
    local tag="$1"; shift
    local desc="$1"; shift
    if check_completed "$tag"; then echo "[skip] $tag already completed"; return 0; fi
    echo ""
    echo "---------------------------------------------------------------"
    echo "  [$tag]  $desc"
    echo "---------------------------------------------------------------"
    local run_dir="${OUT_BASE}/${tag}"
    [ "$FORCE" -eq 1 ] && [ -d "$run_dir" ] && rm -rf "$run_dir"
    if bash scripts/research_sweep.sh $COMMON --out-dir "$run_dir" "$@" "$DEPTH" 2>&1 | tee -a "$LOGFILE"; then
        echo "[ok] $tag"; mark_completed "$tag"
    else
        echo "[FAIL] $tag"; return 1
    fi
}

# ---- control: dense at its own budget, which pins every other arm ----------
DENSE_PIN=""
_stored="$(get_var tokens_d${DEPTH})"
if [ -n "$_stored" ] && [ "$(get_var tokens_vocab_d${DEPTH})" = "$TOK_VOCAB" ]; then
    DENSE_PIN="--target-tokens $_stored"
fi
run_experiment "DENSE_D${DEPTH}" "Dense control" --use-eet 0 $DENSE_PIN

if [ -z "$_stored" ] && [ -f "$LOGFILE" ]; then
    TOK_N=$(grep -h "Total number of training tokens:" "$LOGFILE" 2>/dev/null | head -1 \
            | sed 's/.*: *//' | tr -d ', ' || true)
    if [ -n "$TOK_N" ]; then
        set_var "tokens_d${DEPTH}" "$TOK_N"; set_var "tokens_vocab_d${DEPTH}" "$TOK_VOCAB"
        _stored="$TOK_N"
    fi
fi
ISO=""; [ -n "$_stored" ] && ISO="--target-tokens $_stored"
echo "[iso-data] all arms pinned to ${_stored:-<unpinned>} tokens"

# ---- T0A: no exits before layer 4 -----------------------------------------
run_experiment "T0A_LATEEXIT_D${DEPTH}" \
    "T0A: --eet-min-exit-layer 4. Layers 1-4 dense, exits only at 5-7." \
    $EET_FLAGS $ISO --eet-min-exit-layer 4 || true

# ---- T0A control: the same config exiting from layer 1, for the profile ----
run_experiment "T0A_CTRL_EARLYEXIT_D${DEPTH}" \
    "T0A control: identical, --eet-min-exit-layer 1. Isolates the exit depth." \
    $EET_FLAGS $ISO --eet-min-exit-layer 1 || true

# ---- T0B: dense, readable at every depth ----------------------------------
run_experiment "T0B_DEEPSUP_D${DEPTH}" \
    "T0B: dense + CE at every layer through the shared head (lambda ${DS_LAMBDA})" \
    --use-eet 0 $ISO \
    --deep-supervision-lambda "${DS_LAMBDA}" --deep-supervision-frac 0.125 || true

# ---- per-layer profile on every checkpoint: THE Tier 0 measurement ---------
echo ""
echo "==============================================================="
echo "  PER-LAYER PROFILES"
echo "==============================================================="
for tag in "DENSE_D${DEPTH}" "T0A_CTRL_EARLYEXIT_D${DEPTH}" "T0A_LATEEXIT_D${DEPTH}" "T0B_DEEPSUP_D${DEPTH}"; do
    CK="${OUT_BASE}/${tag}/depth_${DEPTH}/ckpt_base/base"
    LAST=$(ls "$CK"/model_*.pt 2>/dev/null | sort | tail -1 || true)
    if [ -z "$LAST" ]; then echo "[profile] $tag: no checkpoint, skipped"; continue; fi
    echo ""
    echo "### $tag ###"
    python -m scripts.eet_readout_oracle --ckpt "$LAST" --skip-fit \
        --data-dir "${DATA_DIR:-data}" --tokenizer-dir "${TOKENIZER_DIR:-tokenizer}" \
        ${MAX_SHARDS:+--max-shards $MAX_SHARDS} \
        --out "${OUT_BASE}/profile_${tag}.json" 2>&1 \
        | grep -vE "httpx|huggingface" | sed -n '/PER-LAYER PROFILE/,/DEPTH COLLAPSE\|ARE WORKING/p' || true
done

echo ""
echo "==============================================================="
echo "  Read the verdict off the two lines below, against the criteria"
echo "  at the top of this file. Do not reinterpret them afterwards."
echo "    T0A PASS: last-4-layer gain > 0.05 AND |dx| at L5-7 > 5"
echo "    T0B PASS: deep-sup dense loses < 0.03 bpb vs plain dense"
echo "==============================================================="
grep -h "Minimum validation bpb" "$LOGFILE" 2>/dev/null | tail -4 || true
