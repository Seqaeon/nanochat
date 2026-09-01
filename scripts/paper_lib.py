"""Shared helpers for the MST paper experiment scripts.

The important piece here is `load_any_model`. `nanochat.checkpoint_manager.
build_model` hardcodes `GPT(model_config)` and has no `use_mst` branch, so
loading an MST checkpoint through it builds a dense model and then fails
`load_state_dict`. This module dispatches on `config.use_mst` instead, and loads
strictly so a partially-loaded model can never be silently evaluated.
"""
import os
import sys
import glob

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.gpt import GPT, GPTConfig
from nanochat.mst import MST
from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys
from nanochat.tokenizer import get_tokenizer


def find_last_step(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
    if not files:
        raise FileNotFoundError(f"no model_*.pt in {ckpt_dir}")
    return max(int(os.path.basename(f).split("_")[-1].replace(".pt", "")) for f in files)


def load_any_model(ckpt_dir, device, step=None, tokenizer_dir=None, strict=True):
    """Load a checkpoint as MST or dense GPT, whichever it was trained as.

    Returns (model, tokenizer, config, meta). The model is in eval mode.
    """
    step = find_last_step(ckpt_dir) if step is None else step
    device = torch.device(device) if isinstance(device, str) else device
    model_data, _, meta = load_checkpoint(ckpt_dir, step, device, load_optimizer=False)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    if device.type in {"cpu", "mps"}:
        model_data = {k: (v.float() if v.dtype == torch.bfloat16 else v)
                      for k, v in model_data.items()}

    cfg_kwargs = dict(meta["model_config"])
    _patch_missing_config_keys(cfg_kwargs)
    config = GPTConfig(**cfg_kwargs)

    if getattr(config, "use_mol", False):
        from nanochat.mol import MoL
        cls = MoL
    elif getattr(config, "use_mst", False):
        cls = MST
    else:
        cls = GPT
    with torch.device("meta"):
        model = cls(config)
    model.to_empty(device=device)
    model.init_weights()          # needed for the rotary buffers
    missing, unexpected = [], []
    try:
        model.load_state_dict(model_data, strict=strict, assign=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"state_dict mismatch loading {ckpt_dir} step {step} as "
            f"{cls.__name__}. Refusing to evaluate a partially-loaded model. "
            f"Original error:\n{e}") from e
    model.eval()
    tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)
    print(f"[paper_lib] loaded {cls.__name__} from {ckpt_dir} step {step}: "
          f"L={config.n_layer} D={config.n_embd}"
          + (f" N={config.mst_n_subs} d={config.mst_sub_dim}" if cls is MST else "")
          + (f" {config.mol_n_shared}+{config.mol_topk}of{config.mol_n_blocks}"
             f" d_thin={config.mol_thin_dim}" if cls.__name__ == "MoL" else ""))
    if cls is MST:
        # The checkpoint stores asdict(config), so these are the flags the model
        # was actually trained with. Print them: a mismatch here means every
        # probe below is measuring a different model than the one you trained,
        # and strict loading will not catch it because these flags add no
        # parameters.
        flags = ("mst_multi_scale_windows", "mst_transition_mode",
                 "mst_transition_width_mult", "mst_grad_equalize",
                 "mst_block_diagonal_muon", "mst_sub_lr_scale",
                 "mst_input_mode", "mst_final_mode")
        print("[paper_lib] checkpoint MST flags: "
              + ", ".join(f"{f.replace('mst_', '')}={getattr(config, f, '?')}"
                          for f in flags))
        print(f"[paper_lib] per-stream windows: {getattr(model, 'sub_window_sizes', None)}")
    return model, tokenizer, config, meta


def build_val_batches(tokenizer, batch_size, seq_len, device, data_dir=None,
                      max_shards=None):
    """Validation batch iterator matching scripts/base_train.py."""
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, batch_size, seq_len, split="val", device=device,
        data_dir=data_dir, max_shards=max_shards,
    )


def mst_layers(model):
    """The per-layer modules of an MST model, batched path or legacy."""
    return list(model.layers)


def gpu_peak_tflops(dtype=torch.bfloat16):
    """Advertised peak for MFU. Extend the table for other cards."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0).lower()
    table = {
        "h200": 989.0, "h100": 989.0, "a100": 312.0,
        "l40": 362.0, "4090": 165.2, "3090": 71.0, "a6000": 155.0,
    }
    for k, v in table.items():
        if k in name:
            return v
    return None
