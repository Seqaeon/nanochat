"""Evaluator-owned batch normalization for supported benchmark datasets."""

from __future__ import annotations

import torch


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def prepare_batch(batch, device: torch.device):
    batch = _move_to_device(batch, device)
    input_ids = batch["input_ids"].long()
    targets = batch["labels"].long()
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = input_ids != 0
    return input_ids, targets, attention_mask.bool(), batch.get("target_positions")
