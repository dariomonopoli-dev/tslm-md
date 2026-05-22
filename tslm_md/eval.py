"""Evaluation metrics for TSLM-MD.

Primary:   Pearson r between predicted affinity and PDBbind ground truth
           on held-out PDB ids.
Secondary: abstention precision/recall — did we abstain on the cases that
           would have had the worst prediction error?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class EvalResult:
    n_total: int
    n_parsed: int
    n_confirmed: int
    n_inconclusive: int
    pearson_r_all: float | None
    pearson_r_confirmed: float | None
    mae_confirmed: float | None
    abstention_rate: float
    abstention_precision: float | None  # of abstained, what fraction was in worst-quartile of error?
    abstention_recall: float | None     # of worst-quartile errors, what fraction did we abstain on?

    def as_table(self) -> str:
        lines = [
            f"  n total              = {self.n_total}",
            f"  n parsed             = {self.n_parsed}",
            f"  n CONFIRMED          = {self.n_confirmed}",
            f"  n INCONCLUSIVE       = {self.n_inconclusive}",
            f"  abstention rate      = {self.abstention_rate:.1%}",
            f"  Pearson r (all)      = {self.pearson_r_all!r}",
            f"  Pearson r (confirmed)= {self.pearson_r_confirmed!r}",
            f"  MAE       (confirmed)= {self.mae_confirmed!r}",
            f"  abstention precision = {self.abstention_precision!r}",
            f"  abstention recall    = {self.abstention_recall!r}",
        ]
        return "\n".join(lines)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def evaluate(reports: Iterable, ground_truth: dict[str, float]) -> EvalResult:
    """Compute eval metrics over a collection of Reports.

    Args:
        reports: iterable of tslm_md.agent.Report
        ground_truth: {pdb_id: true_affinity_kcal_mol}
    """
    reports = list(reports)
    n_total = len(reports)
    parsed = [r for r in reports if r.affinity is not None]
    confirmed = [r for r in parsed if r.verdict == "CONFIRMED"]
    inconclusive = [r for r in parsed if r.verdict == "INCONCLUSIVE"]

    abstention_rate = (len(inconclusive) / len(parsed)) if parsed else 0.0

    # Pearson over all parsed (regardless of verdict) and over confirmed-only
    y_true_all = [ground_truth[r.pdb_id] for r in parsed if r.pdb_id in ground_truth]
    y_pred_all = [r.affinity for r in parsed if r.pdb_id in ground_truth]

    y_true_conf = [ground_truth[r.pdb_id] for r in confirmed if r.pdb_id in ground_truth]
    y_pred_conf = [r.affinity for r in confirmed if r.pdb_id in ground_truth]

    pearson_all = _pearson(y_true_all, y_pred_all)
    pearson_conf = _pearson(y_true_conf, y_pred_conf)
    mae_conf = (
        float(np.mean(np.abs(np.asarray(y_true_conf) - np.asarray(y_pred_conf))))
        if y_true_conf else None
    )

    # Abstention precision/recall: define "worst-quartile of error" using parsed reports
    abstention_precision = None
    abstention_recall = None
    if parsed and y_true_all:
        errs = np.abs(np.asarray(y_pred_all) - np.asarray(y_true_all))
        if errs.size >= 4:
            q75 = np.quantile(errs, 0.75)
            is_worst = errs >= q75
            is_abstained = np.array([r.verdict == "INCONCLUSIVE" for r in parsed if r.pdb_id in ground_truth])
            tp = int(((is_worst) & (is_abstained)).sum())
            fp = int(((~is_worst) & (is_abstained)).sum())
            fn = int(((is_worst) & (~is_abstained)).sum())
            abstention_precision = tp / (tp + fp) if (tp + fp) else None
            abstention_recall = tp / (tp + fn) if (tp + fn) else None

    return EvalResult(
        n_total=n_total,
        n_parsed=len(parsed),
        n_confirmed=len(confirmed),
        n_inconclusive=len(inconclusive),
        pearson_r_all=pearson_all,
        pearson_r_confirmed=pearson_conf,
        mae_confirmed=mae_conf,
        abstention_rate=abstention_rate,
        abstention_precision=abstention_precision,
        abstention_recall=abstention_recall,
    )
