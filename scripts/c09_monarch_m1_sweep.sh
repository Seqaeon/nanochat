#!/usr/bin/env bash
# ============================================================================
# C09: sweep the Monarch inner factor m1, which is the grouped GEMM's K.
#
# WHAT WAS ALREADY RUN, AND WHY IT IS NOT THIS
#   c05 swept M and touched m1 exactly once, downward:
#
#     arm                 M     m1 (=K)   m2   block_out    bpb (V=32k, d4)
#     MON_M1024_m1_8   1024           8  128         256    1.3802
#     MON_M256          256          16   16        2048    1.3620
#     MON_M1024        1024          32   32        1024    1.2559
#
#   Every point had K in {8, 16, 32}. m1=8 cost 0.124 bpb against m1=32, which
#   says the UP direction is worth testing, and it never was.
#
# WHY IT MATTERS NOW
#   At V=131072 depth 8 the Monarch arm won on FLOPs (0.364x dense, gap +0.1111
#   against a break-even of +0.1611) and then failed to convert it: MFU 13.74%
#   against dense's 36.06%, so a 2.75x FLOP reduction bought 4% of wall clock.
#
#   The cause is K. `torch.einsum` lowers this to a batched GEMM with K = m1 = 32,
#   which tensor cores cannot fill, and it got worse with vocabulary because
#   block_out grows with V (1024 at 32k, 4096 at 131k) while K stayed put.
#   Recomputed from the c05 numbers, Monarch's MFU was ALREADY 0.60x dense at
#   V=32768 depth 4. This was never a regression; it was invisible.
#
# THE TRADE, AT THE MEASURED V=131072 DENSE SLOPE OF 0.3672 bpb/decade
#   M is held at 1024 throughout so the rank ceiling stays d+1 = 513 and every
#   arm remains the full-rank head c06 validated. Only the factorisation moves.
#
#     m1    m2   block_out   model vs dense   break-even gap   if gap holds at 0.1111
#     32    32        4096           0.364x          +0.1611   WINS
#     64    16        8192           0.407x          +0.1434   WINS
#    128     8       16384           0.492x          +0.1130   WINS, barely
#    256     4       32768           0.664x          +0.0654   LOSES
#
#   So m1=128 is the ceiling on the FLOPs axis, and the question is whether MFU
#   recovers enough before then to make the wall clock win too. If the gap also
#   SHRINKS with m1 (which m1=8 costing 0.124 bpb suggests it might), the
#   break-even column moves and m1=256 comes back into play.
#
# EVERY ARM SHARES ONE DEVICE BATCH, ON PURPOSE
#   This sweep is about wall clock, and wall clock is not comparable across
#   sweeps or across device-batch settings. The Monarch head needs roughly twice
#   dense's head memory (an unavoidable `clone`: bmm returns (m2, N, block_out)
#   and the caller needs (N, m2, block_out), and those orders cannot share
#   memory), so DEVICE_BATCH_SIZE is set low enough for the heaviest arm and the dense
#   anchor uses the same value. Gradient accumulation holds the total batch, so
#   dt covers the same tokens for every arm.
#
#   bash scripts/c09_monarch_m1_sweep.sh              # V=131072 depth 8
#   DEVICE_BATCH_SIZE=16 bash scripts/c09_monarch_m1_sweep.sh
#   M1S="32 128" bash scripts/c09_monarch_m1_sweep.sh --sch-only
#   VOCAB=32768 TOKENIZER_DIR=tokenizer bash scripts/c09_monarch_m1_sweep.sh 4
# ============================================================================
set -o pipefail

FORCE=0
SEEDS=1
RUN_DENSE=1
RUN_SCH=1
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)      FORCE=1; shift ;;
        --seeds)      SEEDS="$2"; shift 2 ;;
        --dense-only) RUN_DENSE=1; RUN_SCH=0; shift ;;
        --sch-only|--mon-only) RUN_DENSE=0; RUN_SCH=1; shift ;;
        [0-9]*)       DEPTHS+=("$1"); shift ;;
        *) echo "unknown arg: $1"
           echo "usage: $0 [--force] [--seeds N] [--dense-only|--sch-only] [DEPTH ...]"
           exit 1 ;;
    esac
done
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(8)

M="${M:-1024}"
M1S="${M1S:-32 64 128 256}"
VOCAB="${VOCAB:-131072}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
OUT_BASE="${OUT_BASE:-out/c09_monarch_m1}"
RANK_CONTEXTS="${RANK_CONTEXTS:-16384}"
# One value for every arm. Halved from the 64 that fits dense, because the
# Monarch head holds two full (N, V) tensors where dense holds one.
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"

if [ "$VOCAB" -eq 131072 ]; then
    TOK="${TOKENIZER_DIR:-${TOKENIZER_DIR_131K:-tokenizer_131k}}"
    if ! python3 -m scripts.ensure_tokenizer --vocab-size 131072 --tokenizer-dir "$TOK" \
            --data-dir "${DATA_DIR:-data}" ${MAX_SHARDS:+--max-shards "$MAX_SHARDS"}; then
        echo "could not prepare the tokenizer at '${TOK}'; nothing was run."
        exit 1
    fi
else
    TOK="${TOKENIZER_DIR:-tokenizer}"
fi
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

for DEPTH in "${DEPTHS[@]}"; do

MODEL_DIM=$(( ((DEPTH * ASPECT_RATIO + 127) / 128) * 128 ))
if [ "$M" -lt "$MODEL_DIM" ]; then
    echo "REFUSING depth ${DEPTH}: M=${M} < d=${MODEL_DIM} would drop the rank ceiling"
    echo "  below d+1 and stop this being the full-rank head c06 validated."
    continue
fi

LOGFILE="${SWEEP_LOG:-${OUT_BASE}/c09_d${DEPTH}.log}"
STATE="${OUT_BASE}/c09_state_d${DEPTH}.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

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

PROBE="--sch-phi-dtype fp32 --sch-rank-probe $RANK_CONTEXTS --sch-bias 1"

run() {
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "SKIP  $t (already completed)"; continue; fi
        echo ""
        echo "--- $t  (depth $depth, V=${VOCAB}, device-batch ${DEVICE_BATCH_SIZE}) ---"
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
echo "  C09: Monarch m1 sweep, V=${VOCAB}, depth ${DEPTH}, d=${MODEL_DIM}, M=${M}"
echo "  m1 values: ${M1S}    device-batch ${DEVICE_BATCH_SIZE} (same for every arm)"
echo "  target tokens ${TARGET_TOKENS}"
echo "  out ${OUT_BASE}"
echo "============================================================"

[ "$RUN_DENSE" -eq 1 ] && run BASE_dense "$DEPTH" --models base --sch-rank-probe $RANK_CONTEXTS
if [ "$RUN_SCH" -eq 1 ]; then
    for m1 in $M1S; do
        run "MON_M${M}_m1_${m1}" "$DEPTH" --models base --use-code-head 1 \
            --sch-head-type monarch --sch-max-m "$M" --sch-monarch-m1 "$m1" $PROBE
    done
fi

done

echo ""
echo "============================================================"
echo "  C09 complete."
echo ""
echo "    python -m scripts.sweep_report ${OUT_BASE}/d<DEPTH>"
echo ""
echo "  READ THE MFU COLUMN FIRST. That is what this sweep exists for."
echo "    dense at V=131072 depth 8:  36.06%"
echo "    m1=32 (the arm that won on FLOPs and not on time):  13.74%"
echo "  If MFU does not climb with m1, K was not the bottleneck and the"
echo "  remaining suspect is the unavoidable full-width clone in the head."
echo ""
echo "  THEN THE TRADE. Break-even gap at the measured 0.3672 bpb/decade:"
echo "    m1= 32  model 0.364x dense  break-even +0.1611   measured gap +0.1111"
echo "    m1= 64  model 0.407x dense  break-even +0.1434"
echo "    m1=128  model 0.492x dense  break-even +0.1130"
echo "    m1=256  model 0.664x dense  break-even +0.0654"
echo "  Larger m1 costs FLOPs, so it has to buy either MFU or bpb to be worth it."
echo "  m1=8 cost 0.124 bpb against m1=32 in c05, so the gap may well shrink"
echo "  going up, which would move the break-even column in Monarch's favour."
echo ""
echo "  WHAT WOULD END THIS: MFU flat across m1, and the gap flat too. Then the"
echo "  head is memory bound on the clone, not compute bound on K, and no choice"
echo "  of factorisation fixes it. The fix would be a fused grouped GEMM that"
echo "  writes its output transposed."
echo "============================================================"
