"""
Smoke test: Dense d16 vs MST SP2_k1 d16, WITH VALUE EMBEDS EXCLUDED.

Dense GPT has value_embeds baked in by default (alternating layers).
There is no --no-ve flag. So "without value embeds" means:
  total_no_ve  = total_params  - ve_params
  active_no_ve = active_params - ve_params
  (VE are lookups, not matmuls — already excluded from FLOPs by estimate_flops())

MST SP2_k1 uses mst_per_stream_ve=0, so its ve_params are the same shared d-wide
tables that Dense uses (not N*d-wide per-stream tables).

SP2_k1 at D16 flags (mirrors D16_k1_ve_plain arm of p08 --group sparse):
  mst_config(SUB_DIM, N_SUBS=4) +
  --mst-sub-head-dim 64  --mst-per-stream-ve 0  --mst-compose-windows 1
  --mst-wo-mode dense    --mst-stream-topk 1    --mst-stream-router-noise 1.0
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.mst import MST

DEPTH        = 16
ASPECT_RATIO = 64
HEAD_DIM     = 128
N_SUBS       = 4
VOCAB_SIZE   = 32768
SEQ_LEN      = 2048
TARGET_RATIO = 10.5

MODEL_DIM = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
SUB_DIM   = MODEL_DIM // N_SUBS
N_HEADS   = MODEL_DIM // HEAD_DIM

print(f"D={DEPTH}  model_dim={MODEL_DIM}  sub_dim={SUB_DIM}  N={N_SUBS}  n_heads={N_HEADS}")
print()


def num_scaling_params(model):
    counts = model.num_scaling_params()
    return counts["transformer_matrices"] + counts["lm_head"]


def base_config():
    return dict(sequence_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                n_layer=DEPTH, n_head=N_HEADS, n_kv_head=N_HEADS, n_embd=MODEL_DIM)


def build_dense():
    cfg = GPTConfig(**base_config())
    with torch.device("meta"):
        m = GPT(cfg)
    return m


def build_mst_sp2_k1_no_ve():
    cfg = GPTConfig(**base_config(),
        use_mst=True, mst_n_subs=N_SUBS, mst_sub_dim=SUB_DIM,
        mst_head_dim=0, mst_input_mode='learned_proj',
        mst_routing_mode='soft_weighted', mst_routing_topk=0, mst_ffn_mode='standard',
        mst_transition_mode='aggregate_distribute', mst_final_mode='concat_proj', mst_final_topk=0,
        mst_routing_aux_weight=0.01, mst_diversity_weight=0.0,
        mst_grad_equalize=1, mst_block_diagonal_muon=1,
        mst_transition_width_mult=float(N_SUBS), mst_sub_lr_scale=2.0, mst_multi_scale_windows=1,
        mst_sub_head_dim=64, mst_per_stream_ve=0, mst_compose_windows=1,
        mst_wo_mode='dense', mst_stream_topk=1, mst_stream_router_noise=1.0,
    )
    with torch.device("meta"):
        m = MST(cfg)
    return m


def stats(model, label):
    total_flops, active_flops, active_params = model.estimate_flops()
    total_params = sum(p.numel() for p in model.parameters())
    ve_params    = sum(ve.weight.numel() for ve in model.value_embeds.values())

    # "Without VE" counts = subtract lookup tables from total/active
    total_no_ve  = total_params  - ve_params
    active_no_ve = active_params - ve_params

    scaling  = num_scaling_params(model)
    n_tokens = int(scaling * TARGET_RATIO)

    # FLOPs: estimate_flops() already excludes VE from matmul FLOPs.
    # total_flops and active_flops are per-sequence → divide by SEQ_LEN for per-token.
    flops_per_token      = active_flops / SEQ_LEN
    total_training_flops = 6 * flops_per_token * n_tokens

    print(f"{'─'*65}")
    print(f"  {label}")
    print(f"  total_params       : {total_params:>20,}  ({total_params:.6e})")
    print(f"  ve_params (lookup) : {ve_params:>20,}")
    print(f"  total_no_ve        : {total_no_ve:>20,}  ({total_no_ve:.6e})  ← report this")
    print(f"  active_params      : {active_params:>20,}  ({active_params:.6e})")
    print(f"  active_no_ve       : {active_no_ve:>20,}  ({active_no_ve:.6e})  ← report this")
    print(f"  active_flops/token : {flops_per_token:.6e}")
    print(f"  scaling_params     : {scaling:>20,}")
    print(f"  n_tokens (Chin.)   : {n_tokens:>20,}")
    print(f"  training_FLOPs     : {total_training_flops:.6e}")
    print()

    return dict(
        total=total_no_ve,
        active=active_no_ve,
        flops=flops_per_token,
        total_training_flops=total_training_flops,
    )


dense_model = build_dense()
mst_model   = build_mst_sp2_k1_no_ve()

d = stats(dense_model, f"Dense  L={DEPTH}")
m = stats(mst_model,   f"MST SP2_k1  L={DEPTH}  (no per-stream VE, G3=OFF)")

print("=" * 65)
print("// JS OBJECT SNIPPETS  (total/active EXCLUDE value_embed lookup tables)")
print("// bpp: null → fill from the actual run log")
print()
print(
    f"  {{ id: 'dense_d{DEPTH}', label: 'Dense d{DEPTH}',"
    f" total: {d['total']},"
    f" active: {d['active']},"
    f" flops: {d['flops']:.6e},"
    f" total_training_flops: {d['total_training_flops']:.6e},"
    f" bpp: null, color: '#4A90D9' }},"
)
print()
print(
    f"  {{ id: 'mst_sp2_k1_d{DEPTH}_no_ve',"
    f" label: 'MST SP2_k1 d{DEPTH} (no VE)',"
    f" total: {m['total']},"
    f" active: {m['active']},"
    f" flops: {m['flops']:.6e},"
    f" total_training_flops: {m['total_training_flops']:.6e},"
    f" bpp: null, color: '#F5A623' }},"
)
