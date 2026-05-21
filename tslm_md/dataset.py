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
from datasets import Dataset

# OpenTSLM is installed via pip install -e third_party/OpenTSLM
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt

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
        """Load train/val/test splits as HF Datasets.

        Case-insensitive join over splits, featurized.h5, and targets.json.
        MISATO HDF5 + Zenodo splits use uppercase; targets.json keys are lowercase.
        Each row carries pdb_id (lowercase canonical, for targets lookup) and
        pdb_id_h5 (the actual case used in featurized.h5).
        """
        with self.targets_json.open() as f:
            targets = json.load(f)
        targets_lower = {k.lower(): v for k, v in targets.items()}

        with h5py.File(self.featurized_h5, "r") as f:
            h5_keys_lower = {k.lower(): k for k in f.keys()}

        def _load_one(split_filename: str) -> Dataset:
            split_path = self.splits_dir / split_filename
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Split file {split_path} missing — run scripts/preprocess_features.py first."
                )
            with split_path.open() as f:
                raw_ids = [line.strip() for line in f if line.strip()]
            rows = []
            for pid in raw_ids:
                key = pid.lower()
                actual_h5_pid = h5_keys_lower.get(key)
                target = targets_lower.get(key)
                if actual_h5_pid is None or target is None:
                    continue
                rows.append({"pdb_id": key, "pdb_id_h5": actual_h5_pid, **target})
            if self.max_samples and len(rows) > self.max_samples:
                rows = rows[: self.max_samples]
            return Dataset.from_list(rows)

        train = _load_one("train.txt")
        val = _load_one("val.txt")
        test = _load_one("test.txt")
        return train, val, test

    def _get_text_time_series_prompt_list(self, row) -> list[TextTimeSeriesPrompt]:
        """Pair each channel's 30-frame trajectory with its descriptive label.

        Returns one TextTimeSeriesPrompt per channel; order must match
        tslm_md.featurize so the encoder receives consistent label/channel pairs.
        """
        h5_key = row.get("pdb_id_h5") or row["pdb_id"]
        with h5py.File(self.featurized_h5, "r") as f:
            feats = f[h5_key][:]  # [6, 30] float32
        labels = channel_descriptors()
        return [
            TextTimeSeriesPrompt(label, channel.tolist())
            for label, channel in zip(labels, feats)
        ]

    def _get_pre_prompt(self, row) -> str:
        return build_prompts(row["pdb_id"])[0]

    def _get_post_prompt(self, row) -> str:
        return build_prompts(row["pdb_id"])[1]

    def _get_answer(self, row) -> str:
        return row["answer"]
