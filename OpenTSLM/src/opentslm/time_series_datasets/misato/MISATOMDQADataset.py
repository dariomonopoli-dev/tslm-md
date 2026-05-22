"""QADataset subclass for MISATO MD trajectories.

Per-channel univariate float time series at 100 frames per system. v1 ships
4 channels (rmsd_ligand, interaction_energy, distance, bSASA); v2 ships 12
(adds pocket_rmsd, ligand_rgyr, min_contact_distance, n_contacts_4A,
n_polar_contacts_35A, n_hydrophobic_contacts_45A, ligand_internal_rmsd,
com_dist_velocity). With OPENTSLM_MISATO_ADD_DELTAS=1, first-difference Δ
channels are appended, doubling the channel count.

Channel descriptions are looked up from CHANNEL_DESCRIPTIONS by name,
falling back to a generic template. The base instruction text adapts to
the channel count present.

Answer string is a templated rationale ending in "Answer: <pK>". Trajectory
flags (dissociated, unstable, multi_ligand, ligand_drift) are surfaced in
the pre-prompt so the model can condition on them.

Each sample also carries pK + aux multi-task labels (dissociated, drift,
bsasa_drift, label_source) that `OpenTSLMSP.compute_loss` reads when
extra heads are enabled.
"""
from __future__ import annotations

from typing import List, Literal, Tuple

import torch
from datasets import Dataset

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.misato.misato_loader import load_misato_splits

CHANNEL_DESCRIPTIONS = {
    "rmsd_ligand": "ligand heavy-atom RMSD vs frame 0, in Angstroms",
    "interaction_energy": "protein-ligand interaction energy, in kcal/mol",
    "distance": "protein-ligand center-of-mass distance, in Angstroms",
    "bSASA": "buried solvent-accessible surface area of the ligand, in Angstroms squared",
    # v2 channels
    "pocket_rmsd": "binding-site protein heavy-atom RMSD vs frame 0 (pocket flexibility), in Angstroms",
    "ligand_rgyr": "ligand radius of gyration (ligand compactness), in Angstroms",
    "min_contact_distance": "minimum protein-ligand atomic distance per frame, in Angstroms",
    "n_contacts_4A": "count of protein-ligand atom pairs within 4 A (interface density)",
    "n_polar_contacts_35A": "count of N/O-N/O protein-ligand atom pairs within 3.5 A (hydrogen-bond proxy)",
    "n_hydrophobic_contacts_45A": "count of C-C protein-ligand atom pairs within 4.5 A (hydrophobic contacts)",
    "ligand_internal_rmsd": "ligand internal RMSD after self-alignment (ligand internal flexibility), in Angstroms",
    "com_dist_velocity": "absolute frame-to-frame change in ligand-protein CoM distance (ligand mobility), in Angstroms",
}


def _describe_channel(name: str) -> str:
    if name.startswith("delta_"):
        base = name[len("delta_"):]
        base_desc = CHANNEL_DESCRIPTIONS.get(base, base)
        return f"first-difference (Δ frame-to-frame) of the {base_desc}"
    return CHANNEL_DESCRIPTIONS.get(name, name)


def _base_instruction(n_channels: int) -> str:
    return (
        f"You are given {n_channels} time series from a 100-frame molecular dynamics "
        f"simulation of a protein-ligand complex. The channels summarize different "
        f"aspects of the trajectory (ligand pose stability, binding energy, "
        f"interface contacts, pocket flexibility, ligand mobility, and where "
        f"applicable their frame-to-frame derivatives). Reason step by step about "
        f"what the trajectory shows and estimate the binding affinity as pK "
        f"(negative log of the dissociation constant in molar; higher = stronger binding).\n\n"
        f"Instructions:\n"
        f"- Write a single paragraph that grounds every claim in the channel values.\n"
        f"- Reference absolute values (kcal/mol, Angstroms, Angstroms squared, counts) "
        f"rather than vague language.\n"
        f"- End your response with: Answer: <pK>\n"
    )


class MISATOMDQADataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_misato_splits()

    def _get_answer(self, row) -> str:
        return row["rationale"]

    def _get_pre_prompt(self, row) -> str:
        n_channels = len(row.get("channel_order", []))
        base = _base_instruction(n_channels) if n_channels else _base_instruction(4)
        notes: list[str] = []
        if row["dissociated"]:
            notes.append(
                "Note: the ligand appears to leave its starting pose or drift out of the pocket during the trajectory."
            )
        if row["unstable"]:
            notes.append(
                "Note: the ligand shows a transient large excursion (RMSD > 10 A) before partially recovering."
            )
        if row["multi_ligand"]:
            notes.append(
                "Note: this complex contains multiple ligands; the channels shown describe only the primary ligand."
            )
        if notes:
            return base + "\n" + "\n".join(notes) + "\n"
        return base

    def _get_post_prompt(self, _row) -> str:
        return "Rationale:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        channels = row["channels_norm"]      # (D, 100)
        means = row["channel_means"]
        stds = row["channel_stds"]
        order = row.get("channel_order")
        if order is None:
            # backward compatibility: v1 four channels in fixed order
            order = ["rmsd_ligand", "interaction_energy", "distance", "bSASA"]
        prompts: List[TextTimeSeriesPrompt] = []
        for name, series, mean, std in zip(order, channels, means, stds):
            label = _describe_channel(name)
            text = f"The following is the {label}, with per-system mean {mean:.3f} and std {std:.3f}:"
            prompts.append(TextTimeSeriesPrompt(text, list(series)))
        return prompts

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        # Carry pK + identifiers + aux multi-task labels.
        sample["pK"] = float(row["pK"])
        sample["pdb_id"] = row["pdb_id"]
        sample["dissociated"] = bool(row["dissociated"])
        sample["unstable"] = bool(row["unstable"])
        sample["multi_ligand"] = bool(row["multi_ligand"])
        sample["ligand_drift"] = bool(row.get("ligand_drift", False))
        sample["bsasa_drift"] = float(row.get("bsasa_drift", 0.0))
        sample["label_source"] = row.get("label_source", "unknown")
        return sample


if __name__ == "__main__":
    ds = MISATOMDQADataset(split="train", EOS_TOKEN="")
    val = MISATOMDQADataset(split="validation", EOS_TOKEN="")
    test = MISATOMDQADataset(split="test", EOS_TOKEN="")
    print(f"sizes: train={len(ds)} val={len(val)} test={len(test)}")
    s = ds[0]
    print("keys:", list(s.keys()))
    print("pdb_id:", s["pdb_id"], "pK:", s["pK"])
    print("pre_prompt[:200]:", s["pre_prompt"][:200])
    print("time_series_text:")
    for t in s["time_series_text"]:
        print(" -", t)
    print("answer:", s["answer"])
