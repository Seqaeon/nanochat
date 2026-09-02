#!/usr/bin/env bash
# ============================================================================
# P13: iso-data profile. Every arm sees the SAME tokens, and the baseline is a
# ladder rather than a single point, so a multiplier can be read off directly.
#
# WHY THIS IS THE ONE TO RUN FIRST
#   MST's claim is an inference-cost claim: better quality per FLOP spent at
#   serving time. That is the MoE convention (Mixtral reports 12.9B active of
#   46.7B total), and the standard way to support it is to fix the training
#   data, then compare quality against inference cost.
#
#   Fixing the data removes the objection the compute-optimal ladder invites.
#   Under the ladder's rule MST draws 1.43x the tokens of the dense model
#   matched to it on active FLOPs (p11_active_params.py). Here nobody draws
#   more than anyone else, so the resulting multiplier needs no protocol
#   defence at all.
#
# WHY A DENSE LADDER AND NOT A SINGLE DENSE POINT
#   One dense run at D tokens gives a comparison, not a multiplier: dense L=16
#   will simply beat MST L=16 on bpb while costing 2.76x the FLOPs per token,
#   and neither number alone is the claim. To say "MST reaches this bpb at
#   Nx fewer FLOPs" you need the dense bpb-versus-FLOPs curve AT THIS TOKEN
#   COUNT, which means several dense depths. That is the real price of an
#   unimpeachable comparison and there is no way around it.
#
# THE BUDGET
#   D = 1.168e9 tokens, which is MST L=16's own compute-optimal budget, so that
#   run is already a point and costs nothing to reuse. Four dense runs at that
#   D cost about 4.6e18 total, roughly one L=24 run.
#
# STATE THE PREDICTION BEFORE LOOKING
#   At L=8 iso-token, MST read 1.061x against dense's 1.000x. On its own budget
#   at L=16 it reads 1.222x. The truth is somewhere between and it is NOT the
#   2.4x FLOPs ratio between the two architectures at equal depth: at that
#   ratio MST is a worse model. Expect roughly 1.1-1.2x.
#
#   bash scripts/p13_isodata.sh
#   TOKENS=3657905664 bash scripts/p13_isodata.sh    # MoL L=16's budget
# ============================================================================
set -o pipefail

FORCE=0; SEEDS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

TOKENS="${TOKENS:-1168000000}"
N_SUBS="${N_SUBS:-4}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
DENSE_DEPTHS="${DENSE_DEPTHS:-10 12 14 16}"
MST_DEPTHS="${MST_DEPTHS:-16 20}"
OUT_BASE="${OUT_BASE:-out/p13_isodata}"
mkdir -p "$OUT_BASE"
LOGFILE="${OUT_BASE}/p13.log"
STATE="${OUT_BASE}/p13_state.json"
[ "$FORCE" -eq 1 ] && rm -f "$STATE"
[ -f "$STATE" ] || echo '{"completed":{}}' > "$STATE"

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

COMMON="--device-batch-size ${DEVICE_BATCH_SIZE:-32} --total-batch-size -1 \
  --use-onecycle 0 --log-every ${LOG_EVERY:-200} --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 --warmdown-ratio 0.65 --final-lr-frac 0.05 \
  --research-dim -1 --target-active-params 0 \
  --save-every 200 --eval-every -1 --target-tokens ${TOKENS}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

run() {
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "SKIP $t"; continue; fi
        echo ""; echo "=== $t (depth $depth, ${TOKENS} tokens) ==="
        local dir="${OUT_BASE}/${t}"
        [ "$FORCE" -eq 1 ] && rm -rf "$dir"
        if bash scripts/research_sweep.sh $COMMON --out-dir "$dir" --seed "$s" \
               "$@" "$depth" 2>&1 | tee -a "$LOGFILE"; then
            mark_done "$t"; echo "OK $t"
        else
            echo "FAIL $t"
        fi
    done
}

# Headline SP2_k1, same reasoning as p12: this profile exists to validate the config
# the ladder is built on. S=1 measured Pareto-neutral against it at L=8 and, after
# correcting its 9.75% token excess, at L=16, so it would not change the answer, and
# the control's lower FLOPs/token makes it both cheaper here and better positioned on
# the inference axis this profile is about.
mst_config() {                            # mst_config <depth>
    local D=$(( (($1 * ASPECT_RATIO + 127) / 128) * 128 ))
    local SD=$(( D / N_SUBS ))
    echo "--use-mst 1 --models base --mst-n-subs $N_SUBS --mst-sub-dim $SD \
      --mst-head-dim 0 --mst-input-mode learned_proj \
      --mst-routing-mode soft_weighted --mst-routing-topk 0 --mst-ffn-mode standard \
      --mst-transition-mode aggregate_distribute \
      --mst-final-mode concat_proj --mst-final-topk 0 \
      --mst-routing-aux-weight 0.01 --mst-diversity-weight 0.0 \
      --mst-grad-equalize 1 --mst-block-diagonal-muon 1 \
      --mst-transition-width-mult ${N_SUBS}.0 --mst-sub-lr-scale 2.0 \
      --mst-multi-scale-windows 1 \
      --mst-sub-head-dim 64 --mst-per-stream-ve 1 --mst-compose-windows 1 \
      --mst-wo-mode dense --mst-stream-topk 1 --mst-stream-router-noise 1.0"
}

echo "============================================================"
echo "  P13 iso-data profile   D = ${TOKENS} tokens for every arm"
echo "  dense depths: ${DENSE_DEPTHS}    MST depths: ${MST_DEPTHS}"
echo "  out ${OUT_BASE}"
echo "============================================================"

for d in $DENSE_DEPTHS; do
    run "ISOD_dense_d${d}" "$d" --models base
done
for d in $MST_DEPTHS; do
    SD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / N_SUBS ))
    if [ $(( SD % 64 )) -ne 0 ]; then
        echo "SKIP MST d${d}: sub_dim ${SD} not divisible by mst_sub_head_dim 64"
        continue
    fi
    run "ISOD_mst_d${d}" "$d" $(mst_config "$d")
done

echo ""
echo "============================================================"
echo "  done: ${OUT_BASE}/"
echo "  Fit bpb = a * (active FLOPs/token)^b on the DENSE runs only, then read"
echo "  each MST run's multiplier off that fit. Every arm saw ${TOKENS} tokens,"
echo "  so the fit and the multipliers carry no budget-rule assumption."
echo "============================================================"
