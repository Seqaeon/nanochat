#!/bin/bash
# Phase 23: Tiny-Experts RemixedLinear + Standard MoE Baseline
#
# Experiments:
#
#   ── DENSE BASELINES ────────────────────────────────────────────────────────
#   23_BASE_DENSE      — plain dense transformer (no MoE, no remix)
#   23_REMIX_WEIGHT    — dense RemixedLinear, weight mod, no template bank
#   23_REMIX_HOUSE     — dense RemixedLinear, householder mod, no template bank
#   23_REMIX_CKR       — dense RemixedLinear, CKR (causal kernel) mod
#
#   ── STANDARD MOE (param-parity experts, learned router) ───────────────────
#   23_STD_MOE_TOP1    — StandardMoE K=8, topk=1  (max sparsity)
#   23_STD_MOE_TOP_OPT — StandardMoE K=8, topk=optimal (E^0.7≈5 for E=8, c=0.3)
#   (expert_dim = 4*D // E for both — param parity with dense)
#
#   ── TINY EXPERTS — weight modulation ──────────────────────────────────────
#   23_TINY_WEIGHT_4T_FROZEN   — K_total=64, topk=16, frozen route
#   23_TINY_WEIGHT_4T_LEARNED  — K_total=64, topk=16, learned route
#   23_TINY_WEIGHT_TOP1_FROZEN — K_total=64, topk=1,  frozen route (full expert_dim)
#   23_TINY_WEIGHT_TOP1_LEARNED
#
#   ── TINY EXPERTS — householder modulation ─────────────────────────────────
#   23_TINY_HOUSE_4T_FROZEN    — K_total=64, topk=16, frozen route
#   23_TINY_HOUSE_4T_LEARNED   — K_total=64, topk=16, learned route
#
#   ── TINY EXPERTS — CKR modulation ─────────────────────────────────────────
#   23_TINY_CKR_4T_FROZEN      — K_total=64, topk=16, frozen route
#   23_TINY_CKR_4T_LEARNED     — K_total=64, topk=16, learned route
#
# Key design:
#   expert_dim = basis_size // topk  (compute parity with dense baseline)
#   FFN layers:  per-token routing
#   Attn layers: per-sequence routing (mean-pool x over T)
#
# Usage (from nanochat root):
#   bash scripts/p23_sweep.sh           # run all (resumes via .state file)
#   bash scripts/p23_sweep.sh --force   # re-run everything
#   SWEEP_LOG=p23.log bash scripts/p23_sweep.sh

set -o pipefail  # pipefail: pipeline exit code = first failing command, not tee's

LOGFILE="${SWEEP_LOG:-sweep.log}"
STATEFILE="${LOGFILE%.log}.state"
FORCE=0
if [[ "$1" == "--force" ]]; then
    FORCE=1
    rm -f "$STATEFILE"
    shift
fi

# ─────────────────────────────────────────
# Resume helpers
# ─────────────────────────────────────────
check_completed() {
    local tag="$1"
    if [[ "$FORCE" -eq 1 ]]; then return 1; fi
    if [[ ! -f "$STATEFILE" ]]; then return 1; fi
    grep -qx "$tag" "$STATEFILE" 2>/dev/null && return 0 || return 1
}

mark_completed() {
    echo "$1" >> "$STATEFILE"
}

print_header() {
    local num="$1" tag="$2" desc="$3"
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  [$num]  $tag"
    echo "║  $desc"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
}

# ─────────────────────────────────────────
# Shared settings
# ─────────────────────────────────────────
DEPTH=4

# Dense base-model flags
BASE_COMMON="--fp8 --max-shards 170 --models base \
  --device-batch-size 8 --use-onecycle 0 --log-every 1 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 --mu-p-mode base_only \
  --warmup-ratio 0.15 \
  --research-dim -1"

# RemixedLinear flags — context conditioning ENABLED with SharedContextGates
# (batches both FFN layer gate MLPs into 3 shared matmuls per block; attn layers use local gates).
# Half the device-batch-size vs dense to give MoE layers VRAM headroom.
REMIX_COMMON="--fp8 --max-shards 170 --models remixed-linear \
  --device-batch-size 2 --use-onecycle 0 --log-every 1 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --warmup-ratio 0.20 \
  --warmdown-ratio 0.50 \
  --research-dim -1 \
  --remix-use-context 1 \
  --cclblock-gate-temperature 2.0 \
  --remix-shared-context-gates 1"

# ══════════════════════════════════════════════════════
# 1: Dense baseline — anchor reference
# ══════════════════════════════════════════════════════
TAG="23_BASE_DENSE"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1" "$TAG" "Dense baseline (plain transformer, no MoE)"
    if bash scripts/research_sweep.sh $BASE_COMMON \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

# ══════════════════════════════════════════════════════
# 1B: RemixedLinear, Linear gate, COMPRESSED basis B=C//4
#     (basis_scale_factor=4, the default)
# ══════════════════════════════════════════════════════
CCL_MOD="${CCL_MOD:-weight}"
CCL_STREAM="${CCL_STREAM:-selective}"

TAG="23_REMIX_${CCL_MOD^^}_LinearGate_C4"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1B" "$TAG" "RemixedLinear, linear gate, B=C//4, temp=2.0"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation $CCL_MOD \
      --cclblock-context-stream $CCL_STREAM \
      --p22-n-templates 1 \
      --remix-use-context 1 \
      --remix-shared-context-gates 0 \
      --remix-use-basis-gate 1 \
      --remix-use-output-gate 1 \
      --remix-basis-gate-mode linear \
      --remix-basis-scale-factor 4 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi
# ══════════════════════════════════════════════════════
# 1E: RemixedLinear, Linear gate, FULL RANK basis B=C
#     Directly mirrors the depth-12 run in sweep_p23.log
#     (basis_scale_factor=1 → min(in,out)//1 = min(in,out))
# ══════════════════════════════════════════════════════
TAG="23_REMIX_${CCL_MOD^^}_LinearGate_FullRank"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1E" "$TAG" "RemixedLinear, linear gate, B=C (full rank), temp=2.0"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation $CCL_MOD \
      --cclblock-context-stream $CCL_STREAM \
      --p22-n-templates 1 \
      --remix-use-context 1 \
      --remix-shared-context-gates 0 \
      --remix-use-basis-gate 1 \
      --remix-use-output-gate 1 \
      --remix-basis-gate-mode linear \
      --remix-basis-scale-factor 1 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi
# ══════════════════════════════════════════════════════
# 1F: RemixedLinear, CENTERED gate, full rank B=C
#     Fix for the 277x gradient disparity:
#     1+tanh(s*logits) starts at 1.0 (passthrough) not 0.5.
#     W_b & W_m train like dense for first ~100 steps, then
#     gate gradually learns to amplify/suppress symmetrically.
# ══════════════════════════════════════════════════════
TAG="23_REMIX_${CCL_MOD^^}_CenteredGate_FullRank"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1F" "$TAG" "RemixedLinear, centered gate, B=C (full rank), temp=2.0"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation $CCL_MOD \
      --cclblock-context-stream $CCL_STREAM \
      --p22-n-templates 1 \
      --remix-use-context 1 \
      --remix-shared-context-gates 0 \
      --remix-use-basis-gate 1 \
      --remix-use-output-gate 1 \
      --remix-basis-gate-mode centered \
      --remix-basis-scale-factor 4 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi


# ══════════════════════════════════════════════════════
# 1G: RemixedLinear, CENTERED gate,  COMPRESSED basis B=C//4
#     Fix for the 277x gradient disparity:
#     1+tanh(s*logits) starts at 1.0 (passthrough) not 0.5.
#     W_b & W_m train like dense for first ~100 steps, then
#     gate gradually learns to amplify/suppress symmetrically.
# ══════════════════════════════════════════════════════
TAG="23_REMIX_${CCL_MOD^^}_CenteredGate_C4"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1G" "$TAG" "RemixedLinear, centered gate, B=C//4, temp=2.0"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation $CCL_MOD \
      --cclblock-context-stream $CCL_STREAM \
      --p22-n-templates 1 \
      --remix-use-context 1 \
      --remix-shared-context-gates 0 \
      --remix-use-basis-gate 1 \
      --remix-use-output-gate 1 \
      --remix-basis-gate-mode centered \
      --remix-basis-scale-factor 4 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi


# ══════════════════════════════════════════════════════
# 1H: RemixedLinear, CENTERED gate,  COMPRESSED basis B=C//4 Rank=32
#     Fix for the 277x gradient disparity:
#     1+tanh(s*logits) starts at 1.0 (passthrough) not 0.5.
#     W_b & W_m train like dense for first ~100 steps, then
#     gate gradually learns to amplify/suppress symmetrically.
# ══════════════════════════════════════════════════════
TAG="23_REMIX_${CCL_MOD^^}_CenteredGate_C4__Rank_32"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1H" "$TAG" "RemixedLinear, centered gate, B=C//4, temp=2.0, Rank =32"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation $CCL_MOD \
      --cclblock-context-stream $CCL_STREAM \
      --p22-n-templates 1 \
      --remix-use-context 1 \
      --remix-shared-context-gates 0 \
      --remix-use-basis-gate 1 \
      --remix-use-output-gate 1 \
      --remix-basis-gate-mode centered \
      --remix-basis-scale-factor 4 \
      --remix-output-gate-rank 32 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi
# ══════════════════════════════════════════════════════
# 1I: Multi-basis C4 — 4 x C//4 templates, learned routing
#     Each of 4 bases has rank C//4. Mixed together they can span
#     up to full rank C. Active FLOPs per token ≈ C4 FLOPs (one
#     template dominates). Total params ≈ FullRank params.
#     Learned routing since it outperforms frozen in prior sweeps.
#     Tests whether routing diversity closes the C4→FullRank gap.
# ══════════════════════════════════════════════════════
TAG="23_REMIX_${CCL_MOD^^}_MultiBasis_C4_4T_Learned"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1I" "$TAG" "RemixedLinear, 4 C4 bases, learned routing — rank recovery via template diversity"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation $CCL_MOD \
      --cclblock-context-stream $CCL_STREAM \
      --p22-n-templates 4 \
      --p22-template-routing-learned 1 \
      --remix-use-context 1 \
      --remix-shared-context-gates 0 \
      --remix-use-basis-gate 1 \
      --remix-use-output-gate 1 \
      --remix-basis-gate-mode centered \
      --remix-basis-scale-factor 4 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi


# ══════════════════════════════════════════════════════
# 1J: Random gate init — C4, linear gate but kept at default
#     kaiming_uniform init (NOT zero-init).
#     All prior gate modes zero-init the projection so gate
#     outputs start at exactly 0.5 or 1.0. Here gate values
#     at step 0 are spread across (0,1) randomly.
#     Tests if a random starting gate hurts, is neutral, or
#     accidentally helps by breaking symmetry from init.
# ══════════════════════════════════════════════════════
TAG="23_REMIX_${CCL_MOD^^}_RandomGateInit_C4"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1J" "$TAG" "RemixedLinear, random gate init (kaiming), B=C//4"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation $CCL_MOD \
      --cclblock-context-stream $CCL_STREAM \
      --p22-n-templates 1 \
      --remix-use-context 1 \
      --remix-shared-context-gates 0 \
      --remix-use-basis-gate 1 \
      --remix-use-output-gate 1 \
      --remix-basis-gate-mode random \
      --remix-basis-scale-factor 4 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi


# ══════════════════════════════════════════════════════
# 1K: Gate LR reduction — C4, centered gate, gate params at
#     0.05× structural LR (down from default 0.3×).
#     Current gate params get 0.3× Muon LR vs structural weights.
#     277× gradient disparity analysis suggests gate params still
#     learn too fast relative to W_b/W_m in C4 context.
#     0.05× forces the gate to move much more slowly, keeping
#     structural weights in the "driver's seat" throughout training.
# ══════════════════════════════════════════════════════
TAG="23_REMIX_${CCL_MOD^^}_GateLR005_CenteredGate_C4"
if check_completed "$TAG"; then
    echo "⏭  Skipping $TAG (already completed)"
else
    print_header "1K" "$TAG" "RemixedLinear, centered gate, B=C//4, gate LR = 0.05× structural LR"
    if bash scripts/research_sweep.sh $REMIX_COMMON \
      --cclblock-modulation $CCL_MOD \
      --cclblock-context-stream $CCL_STREAM \
      --p22-n-templates 1 \
      --remix-use-context 1 \
      --remix-shared-context-gates 0 \
      --remix-use-basis-gate 1 \
      --remix-use-output-gate 1 \
      --remix-basis-gate-mode centered \
      --remix-basis-scale-factor 4 \
      --remix-gate-lr-scale 0.05 \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "════════════════ $TAG COMPLETE ════════════════"
        mark_completed "$TAG"
    else
        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
    fi
fi

# ══════════════════════════════════════════════════════
# 1D: DualGateLinear — commented out (worse than dense, see sweep_p23 (27).log)
# ══════════════════════════════════════════════════════
#TAG="23_DUAL_GATE_${CCL_MOD^^}_Linear"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "1D" "$TAG" "DualGateLinear, ${CCL_MOD} mod, ${CCL_STREAM} stream"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --cclblock-modulation $CCL_MOD \
#      --cclblock-context-stream $CCL_STREAM \
#      --p22-n-templates 1 \
#      --remix-use-context 1 \
#      --remix-shared-context-gates 0 \
#      --remix-use-basis-gate 1 \
#      --remix-use-output-gate 1 \
#      --remix-basis-gate-mode linear \
#      --remix-use-dual-gate 1 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi

# ══════════════════════════════════════════════════════
#TAG="23_REMIX_${CCL_MOD^^}_MLPGate"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "1C" "$TAG" "Dense RemixedLinear, ${CCL_MOD} mod, ${CCL_STREAM} stream"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --cclblock-modulation $CCL_MOD \
#      --cclblock-context-stream $CCL_STREAM \
#      --p22-n-templates 1 \
#      --remix-use-context 1 \
#      --remix-shared-context-gates 0 \
#      --remix-use-basis-gate 1 \
#      --remix-use-output-gate 1 \
#      --remix-basis-gate-mode mlp \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi
# ══════════════════════════════════════════════════════
# 2: TinyExpert K=8, top-1, Quantile Routing
# ══════════════════════════════════════════════════════
#TAG="23_QROUTE_TINY_K8_TOP1"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "2" "$TAG" "TinyExpert K=8, top-1, quantile routing, no context"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p23-tiny-expert 1 \
#      --p23-n-experts 8 \
#      --p23-topk 1 \
#      --p23-quantile-route 2 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi
#
# ══════════════════════════════════════════════════════
# 3: TinyExpert K=64, top-16, Quantile Routing
# ══════════════════════════════════════════════════════
#TAG="23_QROUTE_TINY_K64_TOP16"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "3" "$TAG" "TinyExpert K=64, top-16, quantile routing, no context"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p23-tiny-expert 1 \
#      --p23-n-experts 64 \
#      --p23-topk 16 \
#      --p23-quantile-route 2 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi
#
# ══════════════════════════════════════════════════════
# 4: LoKR K=64, rank=4, top-16 (SKIP - Quantile not tested yet)
# ══════════════════════════════════════════════════════
#TAG="23_LOKR_K64_TOP16"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "4" "$TAG" "LoKR K=64, top-16, rank=4, no context, compile enabled"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p23-lokr 1 \
#      --p23-n-experts 64 \
#      --p23-topk 16 \
#      --p23-lokr-rank 4 \
#      --p23-learned-route 1 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi

# ══════════════════════════════════════════════════════
# 5: LinearMoE K=8, top-1, Quantile Routing
# ══════════════════════════════════════════════════════
#TAG="23_QROUTE_LINEAR_MOE_K8_TOP1"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "5" "$TAG" "LinearMoE K=8, top-1, quantile routing, no context"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p23-linear-moe-experts 8 \
#      --p23-linear-moe-topk 1 \
#      --p23-quantile-route 2 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi
#
# ══════════════════════════════════════════════════════
# 6: LinearMoE K=8, top-16, Quantile Routing
# ══════════════════════════════════════════════════════
#TAG="23_QROUTE_LINEAR_MOE_K8_TOP16"
#if check_completed "$TAG"; then
#    echo "⏭  Skipping $TAG (already completed)"
#else
#    print_header "6" "$TAG" "LinearMoE K=8, top-16, quantile routing, no context"
#    if bash scripts/research_sweep.sh $REMIX_COMMON \
#      --p23-linear-moe-experts 8 \
#      --p23-linear-moe-topk 16 \
#      --p23-quantile-route 2 \
#      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
#        echo "════════════════ $TAG COMPLETE ════════════════"
#        mark_completed "$TAG"
#    else
#        echo "════════════════ $TAG FAILED — will retry next run ════════════════"
#    fi
#fi

# ══════════════════════════════════════════════════════
# ARCHIVED (context-modulation dependent)
# ══════════════════════════════════════════════════════
# 23_REMIX_WEIGHT/HOUSE/CKR, 23_STD_MOE_TOP1/TOP_OPT
# 23_TINY_WEIGHT/HOUSE/CKR variants, 23_LOKR_WEIGHT/HOUSE/CKR

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Phase 23 Sweep Complete (6 experiments)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Check $LOGFILE for results."
