"""MDCoTQADataset — yields the OpenTSLM 5-key dict for binding-affinity prediction.

Subclasses opentslm.time_series_datasets.QADataset. One sample per PDB id:

    {
      "time_series":      Tensor[6, 30],     # featurized trajectory
      "time_series_text": [str],             # textual descriptor of the series
      "pre_prompt":       str,               # task prompt
      "post_prompt":      str,               # answer-format instruction
      "answer":           str,               # "Answer: <x> kcal/mol. Confidence: <y>."
    }

Data sources:
    data/featurized.h5    (written by scripts/preprocess_features.py)
    data/targets.json     (written by scripts/build_training_targets.py)
    data/splits/{train,val,test}.txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Tuple

import h5py
import torch
from datasets import Dataset

# OpenTSLM is installed via pip install -e third_party/OpenTSLM
from opentslm.time_series_datasets.QADataset import QADataset

from tslm_md.prompts import build_prompts, channel_descriptors


class MDCoTQADataset(QADataset):
    """Stage-6 training dataset for protein-ligand binding affinity from MD trajectories."""

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        featurized_h5: str | Path = "data/featurized.h5",
        targets_json: str | Path = "data/targets.json",
        splits_dir: str | Path = "data/splits",
        format_sample_str: bool = False,
        time_series_format_function=None,
        max_samples: int | None = None,
    ):
        self.featurized_h5 = Path(featurized_h5)
        self.targets_json = Path(targets_json)
        self.splits_dir = Path(splits_dir)
        self.max_samples = max_samples

        # OpenTSLM uses "validation" externally and "val" internally in some places —
        # accept the OpenTSLM naming, translate to our split-file name.
        self._split_filename = {
            "train": "train.txt",
            "test": "test.txt",
            "validation": "val.txt",
        }[split]

        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load train/val/test splits as HF Datasets keyed by pdb_id."""
        with self.targets_json.open() as f:
            targets = json.load(f)  # {pdb_id: {"answer": "...", "affinity_kcal_mol": float, "confidence": "high|medium|low"}}

        def _load_one(split_filename: str) -> Dataset:
            split_path = self.splits_dir / split_filename
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Split file {split_path} missing — run scripts/preprocess_features.py first."
                )
            with split_path.open() as f:
                ids = [line.strip() for line in f if line.strip()]
            ids = [i for i in ids if i in targets]  # only ids we have labels for
            if self.max_samples and len(ids) > self.max_samples:
                ids = ids[: self.max_samples]
            rows = [{"pdb_id": pid, **targets[pid]} for pid in ids]
            return Dataset.from_list(rows)

        train = _load_one("train.txt")
        val = _load_one("val.txt")
        test = _load_one("test.txt")
        return train, val, test

    # OpenTSLM QADataset reads time_series from a key on the row dict. We
    # override the per-row generators to inject our featurized tensor + prompts.

    def _get_time_series(self, row) -> torch.Tensor:
        """Read the [6, 30] feature tensor for this PDB id from featurized.h5."""
        pdb_id = row["pdb_id"]
        with h5py.File(self.featurized_h5, "r") as f:
            return torch.from_numpy(f[pdb_id][:])  # [6, 30] float32

    def _get_time_series_text(self, row) -> list[str]:
        # ONE descriptor per channel — pairs with each Chronos-encoded chunk.
        # Order MUST match the channel order in tslm_md.featurize.
        return channel_descriptors()

    def _get_pre_prompt(self, row) -> str:
        return build_prompts(row["pdb_id"])[0]

    def _get_post_prompt(self, row) -> str:
        return build_prompts(row["pdb_id"])[1]

    def _get_answer(self, row) -> str:
        return row["answer"]
