#!/usr/bin/env python3
"""Diagnose template bank diversity and routing effectiveness in RemixedLinear checkpoints.

Loads a trained RemixedLinear checkpoint and computes:
  1. Pairwise cosine similarity between templates (per layer)
  2. Effective rank of the template bank (SVD of the K × (out·basis) matrix)
  3. Template weight distribution from stored buffers
  4. Output gate statistics (if gates are active)

Usage:
    python scripts/remix_diagnostics.py <checkpoint_path>
    python scripts/remix_diagnostics.py out/sweep_p35_ablation/35E_FULL_REMIX_8T_D4/depth_4/ckpt_remixed-linear/remixed-linear
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def cosine_sim_matrix(bank: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between K templates.

    Args:
        bank: (K, *shape) — each template is flattened to a vector.
    Returns:
        (K, K) cosine similarity matrix.
    """
    K = bank.shape[0]
    flat = bank.reshape(K, -1).float()
    flat = F.normalize(flat, dim=-1)
    return flat @ flat.T


def effective_rank(bank: torch.Tensor) -> float:
    """Effective rank via Shannon entropy of normalized singular values.

    eff_rank = exp(-Σ p_i log p_i) where p_i = σ_i / Σ σ_j.
    Ranges from 1 (rank-1) to K (all singular values equal).
    """
    K = bank.shape[0]
    flat = bank.reshape(K, -1).float()
    s = torch.linalg.svdvals(flat)
    s = s[s > 1e-8]
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()


def stable_rank(bank: torch.Tensor) -> float:
    """Stable rank = ||A||_F^2 / ||A||_2^2. Measures how 'spread' the spectrum is."""
    K = bank.shape[0]
    flat = bank.reshape(K, -1).float()
    s = torch.linalg.svdvals(flat)
    if s[0] < 1e-8:
        return 0.0
    return ((s ** 2).sum() / (s[0] ** 2)).item()


def analyse_checkpoint(ckpt_path: str):
    """Load checkpoint and run all diagnostics."""
    ckpt_path = Path(ckpt_path)

    # Find the model checkpoint file
    if ckpt_path.is_dir():
        candidates = list(ckpt_path.glob("model*.pt")) + list(ckpt_path.glob("*.pt"))
        if not candidates:
            print(f"No .pt files found in {ckpt_path}")
            sys.exit(1)
        ckpt_file = sorted(candidates)[-1]  # latest
    else:
        ckpt_file = ckpt_path

    print(f"Loading checkpoint: {ckpt_file}")
    state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)

    # If wrapped in a 'model' key (common in DDP checkpoints)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # Collect all template banks and routing stats
    banks = {}
    weights_bufs = {}
    entropy_bufs = {}
    output_gate_scales = {}
    output_gate_bases = {}

    for name, param in state_dict.items():
        if "template_bank" in name and param.ndim >= 3:
            banks[name] = param
        elif "_template_weights_buf" in name:
            weights_bufs[name] = param
        elif "_template_entropy_buf" in name:
            entropy_bufs[name] = param
        elif "output_gate_scale" in name:
            output_gate_scales[name] = param
        elif "output_gate_basis" in name:
            output_gate_bases[name] = param

    if not banks:
        print("No template_bank parameters found. Is this a K>1 RemixedLinear checkpoint?")
        print("Checking for template_mixing (K=1 mode)...")
        for name, param in state_dict.items():
            if "template_mixing" in name and param.ndim == 2:
                print(f"  {name}: shape={list(param.shape)}")
        return

    # ─── Template Bank Analysis ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TEMPLATE BANK DIVERSITY ANALYSIS")
    print("=" * 80)

    all_cos_means = []
    all_eff_ranks = []
    all_stable_ranks = []

    for name, bank in sorted(banks.items()):
        K = bank.shape[0]
        cos_mat = cosine_sim_matrix(bank)

        # Extract upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
        pairwise_cos = cos_mat[mask]

        cos_mean = pairwise_cos.mean().item()
        cos_std = pairwise_cos.std().item()
        cos_max = pairwise_cos.max().item()
        cos_min = pairwise_cos.min().item()

        erank = effective_rank(bank)
        srank = stable_rank(bank)

        all_cos_means.append(cos_mean)
        all_eff_ranks.append(erank)
        all_stable_ranks.append(srank)

        # Shorten name for display
        short = name.replace("transformer.h.", "L").replace(".template_bank", "")
        short = short.replace(".attn.", ".").replace(".ffwd.", ".ff.")

        print(f"\n  {short} | shape={list(bank.shape)}")
        print(f"    Pairwise cosine sim: mean={cos_mean:.4f}  std={cos_std:.4f}  "
              f"range=[{cos_min:.4f}, {cos_max:.4f}]")
        print(f"    Effective rank: {erank:.2f} / {K}  "
              f"| Stable rank: {srank:.2f} / {K}")

        if cos_mean > 0.8:
            print(f"    ⚠️  HIGH SIMILARITY — templates are near-collapsed")
        elif cos_mean > 0.5:
            print(f"    ⚠️  MODERATE similarity — limited diversity")
        elif cos_mean < 0.1:
            print(f"    ✅  Templates are well-diversified")

    print(f"\n{'─' * 80}")
    print(f"  SUMMARY across {len(banks)} layers:")
    print(f"    Mean pairwise cosine: {sum(all_cos_means)/len(all_cos_means):.4f}")
    print(f"    Mean effective rank:  {sum(all_eff_ranks)/len(all_eff_ranks):.2f} / {K}")
    print(f"    Mean stable rank:     {sum(all_stable_ranks)/len(all_stable_ranks):.2f} / {K}")

    # ─── Template Weight Distribution ───────────────────────────────────────
    if weights_bufs:
        print("\n" + "=" * 80)
        print("TEMPLATE WEIGHT DISTRIBUTION (from _template_weights_buf)")
        print("=" * 80)

        for name, buf in sorted(weights_bufs.items()):
            short = name.replace("transformer.h.", "L").replace("._template_weights_buf", "")
            short = short.replace(".attn.", ".").replace(".ffwd.", ".ff.")
            w = buf.float()
            # Compute usage CV and entropy
            if w.sum() > 0:
                p = w / w.sum()
                ent = -(p * torch.log(p.clamp(min=1e-8))).sum().item()
                max_ent = torch.log(torch.tensor(float(len(w)))).item()
                cv = (w.std() / w.mean()).item() if w.mean() > 0 else float('inf')
                print(f"  {short}: [{', '.join(f'{x:.3f}' for x in w.tolist())}]  "
                      f"H={ent:.3f}/{max_ent:.3f}  CV={cv:.3f}")

    # ─── Output Gate Analysis ───────────────────────────────────────────────
    if output_gate_scales:
        print("\n" + "=" * 80)
        print("OUTPUT GATE SCALE PARAMETERS")
        print("=" * 80)

        for name, scale in sorted(output_gate_scales.items()):
            short = name.replace("transformer.h.", "L").replace(".output_gate_scale", "")
            short = short.replace(".attn.", ".").replace(".ffwd.", ".ff.")
            s = scale.float().item()
            # At init, scale starts small. If it stays small, the gate is near-identity.
            if abs(s) < 0.01:
                status = "⚠️ NEAR-ZERO (gate ≈ identity, not modulating)"
            elif abs(s) < 0.1:
                status = "small (weak modulation)"
            else:
                status = "active"
            print(f"  {short}: scale={s:.6f}  [{status}]")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RemixedLinear template bank diagnostics")
    parser.add_argument("checkpoint", help="Path to checkpoint file or directory")
    args = parser.parse_args()
    analyse_checkpoint(args.checkpoint)
