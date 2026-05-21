"""QADataset subclass for MISATO MD trajectories.

Four univariate channels per system (rmsd_ligand, interaction_energy,
distance, bSASA), each (100,) float. Answer string is a templated rationale
ending in "Answer: <pK>". Trajectory flags (dissociated, unstable,
multi_ligand) are surfaced in the pre-prompt so the model can condition on
them.

For v1b (regression head) every sample also carries a "pK" float field that
`OpenTSLMSP.compute_loss` reads when `regression_enabled` is set.
"""
from __future__ import annotations

from typing import List, Literal, Tuple

import torch
from datasets import Dataset

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.misato.misato_loader import load_misato_splits

CHANNEL_LABELS = [
    "ligand heavy-atom RMSD vs frame 0, in Angstroms",
    "protein-ligand interaction energy, in kcal/mol",
    "protein-ligand center-of-mass distance, in Angstroms",
    "buried solvent-accessible surface area of the ligand, in Angstroms squared",
]

_BASE_INSTRUCTION = """\
You are given four time series from a 100-frame molecular dynamics simulation
of a protein-ligand complex. Each channel summarizes one aspect of the
trajectory: ligand pose stability, binding energy, ligand-protein distance,
and contact surface area. Reason step by step about what the trajectory shows
and estimate the binding affinity as pK (negative log of the dissociation
constant in molar; higher = stronger binding).

Instructions:
- Write a single paragraph that grounds every claim in the four channel values.
- Reference absolute values (kcal/mol, Angstroms, Angstroms squared) rather than vague language.
- End your response with: Answer: <pK>
"""


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
            return _BASE_INSTRUCTION + "\n" + "\n".join(notes) + "\n"
        return _BASE_INSTRUCTION

    def _get_post_prompt(self, _row) -> str:
        return "Rationale:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        channels = row["channels_norm"]  # list of 4 lists, each length 100
        means = row["channel_means"]
        stds = row["channel_stds"]
        prompts: List[TextTimeSeriesPrompt] = []
        for label, series, mean, std in zip(CHANNEL_LABELS, channels, means, stds):
            text = f"The following is the {label}, with per-system mean {mean:.3f} and std {std:.3f}:"
            prompts.append(TextTimeSeriesPrompt(text, list(series)))
        return prompts

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        # Carry pK + identifiers for the regression head and downstream eval.
        sample["pK"] = float(row["pK"])
        sample["pdb_id"] = row["pdb_id"]
        sample["dissociated"] = bool(row["dissociated"])
        sample["unstable"] = bool(row["unstable"])
        sample["multi_ligand"] = bool(row["multi_ligand"])
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
