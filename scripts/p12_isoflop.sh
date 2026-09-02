#!/usr/bin/env bash
# ============================================================================
# P12: isoFLOP profile. Every arm gets the SAME training compute.
#
# WHY THIS EXISTS
#   The compute-optimal ladder sets each arm's token budget from its own
#   parameter count (10.5 x (matrices + lm_head)). That is a defensible
#   protocol but it is a protocol, and it is the one a reviewer will question,
#   because MST draws 1.43x the tokens of the dense model matched to it on
#   active FLOPs (see p11_active_params.py). An isoFLOP profile removes the
#   question entirely: fix C, vary model size, plot final bpb against C. This
#   is Chinchilla's "approach 2" and the default convention in efficiency work.
#
#   Read it as: at a fixed training-compute budget, which architecture's
#   best-performing model is better, and how big is that model?
#
# THE BUDGET
#   C = 6.710562e17 active FLOPs. That is MST L=16's existing training budget to six
#   figures (10.5 x 111,235,072 scaling params x 5.7455e8 active FLOPs/token), so that
#   run is already a point on the profile and costs nothing to reuse. Six new runs at
#   6.71e17 each is 4.0e18 total, about 0.8x of one L=24 run. isoFLOP cost does not
#   depend on which depths you pick, only on how many.
#
#   A second contour at C=4.8e18 (L=24's budget) would reuse MST L=24 the same
#   way, but costs ~6x an L=24 run. Only worth it if this one is favourable.
#
# ACTIVE, NOT TOTAL
#   --target-active-flops, not --target-flops. base_train's --target-flops
#   divides by estimate_flops()[0], the TOTAL count, while the Pareto plots use
#   [1], the active one. For MST those differ by 1.34x at L=24, so using the
#   total would put MST on a tighter contour than dense and bias the profile
#   against it without anyone noticing.
#
# EXPECTED SHAPE OF THE ANSWER
#   MST's training-FLOPs multiplier on the ladder is 1.08-1.14x, and the bpb
#   gap on that axis is 0.003-0.005, so a small separation is the honest
#   prediction. State that before looking. This profile is for credibility,
#   not for a big number; if you want the big number it is on the inference
#   axis and p13_isodata.sh is the run that gets it.
#
#   bash scripts/p12_isoflop.sh
#   FLOPS=4.8e18 bash scripts/p12_isoflop.sh
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

FLOPS="${FLOPS:-6.710562e17}"
N_SUBS="${N_SUBS:-4}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
DENSE_DEPTHS="${DENSE_DEPTHS:-8 10 12 14}"
# MST needs mst_sub_head_dim (64) to divide sub_dim = D/N, i.e. D a multiple of 256.
# L=12,16,24 give D=768,1024,1536 -> d=192,256,384, all divisible by 64. L=18 and L=22
# do NOT (d=288, 352) and have no MST arm at all.
#
# L=20 is deliberately EXCLUDED. It is MST's known off-trend ladder point (1.125x on
# FLOPs/token against 1.222x at L=16 and 1.284x at L=24), so including it would let a
# single suspect measurement pull the isoFLOP frontier down. L=12 and L=24 bracket the
# optimum at this budget from below and above, which is what the profile needs.
MST_DEPTHS="${MST_DEPTHS:-12 16 24}"
OUT_BASE="${OUT_BASE:-out/p12_isoflop}"
mkdir -p "$OUT_BASE"
LOGFILE="${OUT_BASE}/p12.log"
STATE="${OUT_BASE}/p12_state.json"
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
  --research-dim -1 --target-active-params 0 --target-tokens -1 \
  --save-every 200 --eval-every -1 --target-active-flops ${FLOPS}"
[ -n "${MAX_SHARDS:-}" ] && COMMON="$COMMON --max-shards $MAX_SHARDS"

run() {
    local tag="$1"; shift
    local depth="$1"; shift
    for s in $(seq 1 "$SEEDS"); do
        local t="${tag}_s${s}"
        if done_already "$t"; then echo "SKIP $t"; continue; fi
        echo ""; echo "=== $t (depth $depth, C=${FLOPS} active FLOPs) ==="
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

# The MST arm is the paper's headline SP2_k1 config. NOT the S=1 variant: the point of
# this profile is to validate the configuration the ladder is built on, and a profile
# run on a different config validates nothing. The two are Pareto-equivalent anyway
# (measured neutral at L=8 and, after correcting for its token excess, at L=16), and
# the control has the lower FLOPs/token, so it is both the cheaper run and the stronger
# inference-cost position.
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
echo "  P12 isoFLOP profile   C = ${FLOPS} active FLOPs"
echo "  dense depths: ${DENSE_DEPTHS}    MST depths: ${MST_DEPTHS}"
echo "  out ${OUT_BASE}"
echo "============================================================"

for d in $DENSE_DEPTHS; do
    run "ISOF_dense_d${d}" "$d" --models base
done
for d in $MST_DEPTHS; do
    SD=$(( (((d * ASPECT_RATIO + 127) / 128) * 128) / N_SUBS ))
    if [ $(( SD % 64 )) -ne 0 ]; then
        echo "SKIP MST d${d}: sub_dim ${SD} not divisible by mst_sub_head_dim 64"
        continue
    fi
    run "ISOF_mst_d${d}" "$d" $(mst_config "$d")
done

echo ""
echo "============================================================"
echo "  done: ${OUT_BASE}/"
echo "  Every run saw the same ${FLOPS} active training FLOPs, so bpb is"
echo "  directly comparable. Plot bpb against depth per arm; the lower"
echo "  envelope of each arm is that architecture's isoFLOP frontier."
echo "============================================================"
