"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")

def _stack_legacy_templates(state, model):
    """Rebuild `X.template_bank` from a checkpoint that stored templates separately."""
    notes = []
    want = {k for k in model.state_dict() if k.endswith(".template_bank")}
    missing = sorted(want - set(state))
    if not missing:
        return state, notes
    state = dict(state)
    for key in missing:
        prefix = key[: -len("template_bank")]
        target = model.state_dict()[key]
        K, per_template = target.shape[0], target.shape[1:]
        cands = sorted(
            (k for k in state
             if k.startswith(prefix)
             and re.fullmatch(r"(?:template_bank|templates?)[._]\d+(?:\.weight)?",
                              k[len(prefix):])),
            key=lambda k: int(re.findall(r"\d+", k[len(prefix):])[0]),
        )
        if len(cands) == K:
            stacked = torch.stack([state[k] for k in cands], dim=0)
            if stacked.shape != target.shape:
                notes.append(
                    f"SHAPE MISMATCH {key}: checkpoint stacks to {tuple(stacked.shape)}, "
                    f"model wants {tuple(target.shape)}. The rebuilt config disagrees "
                    f"with the checkpoint about basis_size or n_templates.")
                continue
            for k in cands:
                state.pop(k)
            state[key] = stacked
            notes.append(f"stacked {K} legacy tensors -> {key}")
        elif len(cands) == 1 and state[cands[0]].shape == model.state_dict()[key].shape[1:]:
            notes.append(f"FOUND ONLY ONE template matrix for {key}: {cands[0]}")
        else:
            notes.append(f"CANNOT REPAIR {key}: candidates under this module = "
                         + (", ".join(k[len(prefix):] for k in cands) or "(none)"))
    return state, notes

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    device = torch.device(device) if isinstance(device, str) else device
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata — if the file is missing (e.g. accidentally deleted),
    # synthesise a minimal stub so training can resume from the correct step.
    # The model weights and optimizer state are intact; only the dataloader
    # position and EMA loss state will be lost (dataloader restarts from 0).
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
    except FileNotFoundError:
        log0(
            f"WARNING: meta file not found at {meta_path}. "
            f"Synthesising a minimal stub — model weights are intact but "
            f"dataloader position will reset to 0 (some token overlap expected)."
        )
        meta_data = {
            "step": step,
            "val_bpb": None,
            "model_config": {},   # base_train.py doesn't use this field on resume
            "user_config": {},
            "device_batch_size": None,
            "max_seq_len": None,
            "total_batch_size": None,
            "dataloader_state_dict": None,  # causes dataloader to restart from shard 0
            "loop_state": {
                "min_val_bpb": float("inf"),
                "smooth_train_loss": 0.0,
                "total_training_time": 0.0,
            },
        }
        # Write the stub back so subsequent loads don't warn again
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        log0(f"Wrote stub meta to {meta_path} — edit it manually if you have the original values.")
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase, tokenizer_dir=None):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    device = torch.device(device) if isinstance(device, str) else device
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile / DDP issues, which prepend keys with _orig_mod. or module.
    model_data = {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)

    # Trust saved tensors over stored config for which router ran
    ck_has_qrouter = any("_qrouter.route_proj" in k for k in model_data)
    ck_has_plain = any(k.endswith(".template_route") for k in model_data)
    declared = int(model_config_kwargs.get("p23_quantile_route", 0) or 0)
    rlk = dict(model_config_kwargs.get("remixed_linear_kwargs") or {})
    declared_kw = int(rlk.get("use_quantile_route", 0) or 0)

    if ck_has_plain and not ck_has_qrouter and (declared or declared_kw):
        log0(f"Stored config says p23_quantile_route={declared}, but checkpoint has template_route and no _qrouter. Forcing flags to 0.")
        model_config_kwargs["p23_quantile_route"] = 0
        rlk["use_quantile_route"] = 0
        model_config_kwargs["remixed_linear_kwargs"] = rlk
    elif ck_has_qrouter and not (declared or declared_kw):
        log0("Checkpoint contains _qrouter weights but flags say 0. Forcing flags to 1.")
        model_config_kwargs["p23_quantile_route"] = 1
        rlk["use_quantile_route"] = 1
        model_config_kwargs["remixed_linear_kwargs"] = rlk

    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init

    model_data, notes = _stack_legacy_templates(model_data, model)
    for n in notes:
        log0(f"[checkpoint_manager] {n}")

    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None, tokenizer_dir=None):
    if glob.glob(os.path.join(checkpoints_dir, "model_*.pt")):
        checkpoint_dir = checkpoints_dir
    else:
        if model_tag is None:
            try:
                model_tag = find_largest_model(checkpoints_dir)
                log0(f"No model tag provided, guessing model tag: {model_tag}")
                checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
            except FileNotFoundError:
                checkpoint_dir = checkpoints_dir
        else:
            checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
            
        if not glob.glob(os.path.join(checkpoint_dir, "model_*.pt")):
            # Fallback: search recursively inside checkpoints_dir for model_*.pt
            pts = glob.glob(os.path.join(checkpoints_dir, "**", "model_*.pt"), recursive=True)
            if pts:
                candidate_dirs = sorted(set(os.path.dirname(p) for p in pts),
                                        key=lambda d: os.path.getmtime(d), reverse=True)
                checkpoint_dir = candidate_dirs[0]
                log0(f"Found checkpoint directory: {checkpoint_dir}")

    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase, tokenizer_dir=tokenizer_dir)
    return model, tokenizer, meta_data

def load_model(source, device, phase, model_tag=None, step=None, tokenizer_dir=None, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, device, phase, model_tag=model_tag, step=step, tokenizer_dir=tokenizer_dir, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
