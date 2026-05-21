"""Deterministic grounded rationale generator.

Given a [6, F] feature tensor, emit a paragraph whose every claim references
a number actually present in the trajectory. No LLM hallucination surface.

This is the answer to "how do you generate rational labels for molecules?" —
we DO NOT generate fake rationales for training. The LLM is trained on the
minimal answer string only ("Answer: -8.4 kcal/mol. Confidence: high.").
At inference time, the rationale is produced HERE, deterministically, from
the same numbers the model saw — making every sentence auditable.
"""

from __future__ import annotations

from typing import Sequence

import torch

CHANNEL_NAMES = [
    "min_pocket_distance_A",
    "mean_pocket_distance_A",
    "n_close_contacts",
    "ligand_rmsd_A",
    "ligand_radius_of_gyration_A",
    "interface_buriedness",
]


def _trend(series: torch.Tensor) -> str:
    """Cheap monotone/plateau/drift classifier on a 1-D tensor."""
    if series.numel() < 2:
        return "stable"
    n = series.numel()
    first_quarter = series[: max(1, n // 4)].mean().item()
    last_quarter = series[-max(1, n // 4):].mean().item()
    overall_std = series.std().item()
    overall_mean = series.mean().item()
    rel_std = overall_std / (abs(overall_mean) + 1e-6)
    delta = last_quarter - first_quarter
    if rel_std < 0.05:
        return "stable"
    if abs(delta) < 0.5 * overall_std:
        return "fluctuating"
    return "increasing" if delta > 0 else "decreasing"


def deterministic_rationale(feats: torch.Tensor) -> str:
    """Return a single grounded paragraph describing the trajectory.

    Args:
        feats: Tensor[6, F] — the SAME tensor the TSLM consumed.
    """
    if feats.ndim != 2 or feats.shape[0] != 6:
        raise ValueError(f"expected feats shape [6, F], got {tuple(feats.shape)}")

    ch = {name: feats[i] for i, name in enumerate(CHANNEL_NAMES)}

    min_d_start = ch["min_pocket_distance_A"][0].item()
    min_d_end = ch["min_pocket_distance_A"][-1].item()
    min_d_trend = _trend(ch["min_pocket_distance_A"])

    rmsd_mean = ch["ligand_rmsd_A"].mean().item()
    rmsd_trend = _trend(ch["ligand_rmsd_A"])

    contacts_max = ch["n_close_contacts"].max().item()
    contacts_trend = _trend(ch["n_close_contacts"])

    rg_start = ch["ligand_radius_of_gyration_A"][0].item()
    rg_end = ch["ligand_radius_of_gyration_A"][-1].item()
    rg_delta = rg_end - rg_start

    buried_mean = ch["interface_buriedness"].mean().item()

    # Composite: stable bound pose if min_d stable/decreasing, rmsd low+stable, contacts saturated/stable
    pose_signals_good = (
        min_d_trend in ("stable", "decreasing")
        and rmsd_trend in ("stable", "decreasing")
        and rmsd_mean < 2.0
        and contacts_trend in ("stable", "increasing")
    )
    verdict_phrase = (
        "consistent with a stable bound pose"
        if pose_signals_good
        else "consistent with a partially bound or unbinding event"
    )

    sentences = [
        f"Minimum pocket-ligand distance moved from {min_d_start:.2f} Å to {min_d_end:.2f} Å and is {min_d_trend} across frames.",
        f"Ligand RMSD from frame 0 averages {rmsd_mean:.2f} Å and is {rmsd_trend} — {'low and tight' if rmsd_mean < 1.5 else 'moderate' if rmsd_mean < 3.0 else 'high'}.",
        f"Close-contact count peaks at {contacts_max:.0f} pairs and is {contacts_trend}.",
        f"Ligand radius of gyration changed by {rg_delta:+.2f} Å over the trajectory.",
        f"Mean buriedness proxy: {buried_mean:.1f} buried ligand atoms.",
        f"Overall these time-series signals are {verdict_phrase}.",
    ]
    return " ".join(sentences)


def channel_summary_dict(feats: torch.Tensor) -> dict:
    """Structured per-channel summary, useful for the Streamlit demo + verifier."""
    if feats.ndim != 2 or feats.shape[0] != 6:
        raise ValueError(f"expected feats shape [6, F], got {tuple(feats.shape)}")
    out = {}
    for i, name in enumerate(CHANNEL_NAMES):
        s = feats[i]
        out[name] = {
            "start": float(s[0]),
            "end": float(s[-1]),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "trend": _trend(s),
        }
    return out
