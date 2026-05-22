"""Build pitch-ready plots from an eval CSV.

Reads results/eval_*.csv (output of scripts/eval_benchmark.py) and creates:

    - scatter.png      : predicted vs ground truth affinity, coloured by confidence
                         (the headline plot for slide 4)
    - residuals.png    : histogram of (pred - truth) — shows error distribution
    - by_confidence.png: box plot of |error| split by confidence level —
                         demonstrates the abstention is calibrated
    - sparklines.png   : per-channel feature trajectories for the best CONFIRMED
                         and worst (high-error) cases — for demo highlights
    - summary.json     : Pearson, Spearman, MAE, RMSE, n, per-confidence breakdown

Optionally pushes the same plots to an existing wandb run (--wandb-run-id)
so they appear alongside the training loss curves.

Usage:
    python scripts/visualize_predictions.py \\
        --csv results/eval_misato_test_final.csv \\
        --title "MISATO test — final checkpoint" \\
        --output-dir results/vis/misato_test_final \\
        --wandb-run-id p06r0jb5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


CONF_COLOR = {"high": "tab:green", "medium": "tab:orange", "low": "tab:red", None: "tab:grey"}
CONF_ORDER = ["high", "medium", "low"]


def load_eval_csv(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            if r.get("pred") in (None, "", "None"):
                continue
            try:
                rows.append({
                    "pdb_id": r["pdb_id"],
                    "truth": float(r["truth"]),
                    "pred": float(r["pred"]),
                    "confidence": (r.get("confidence") or "").lower() or None,
                    "raw": r.get("raw", ""),
                })
            except (TypeError, ValueError):
                continue
    return rows


def plot_scatter(rows: list[dict], out_path: Path, title: str) -> dict:
    truths = np.array([r["truth"] for r in rows])
    preds = np.array([r["pred"] for r in rows])
    confs = [r["confidence"] for r in rows]

    r_p, _ = pearsonr(preds, truths) if len(preds) >= 2 else (float("nan"), None)
    r_s, _ = spearmanr(preds, truths) if len(preds) >= 2 else (float("nan"), None)
    mae = float(np.mean(np.abs(preds - truths)))
    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=140)
    for conf in CONF_ORDER + [None]:
        mask = np.array([c == conf for c in confs])
        if not mask.any():
            continue
        label = conf if conf else "unspecified"
        ax.scatter(truths[mask], preds[mask], s=24, alpha=0.7,
                   c=CONF_COLOR[conf], edgecolor="white", linewidth=0.4, label=label)

    lo = float(min(truths.min(), preds.min()))
    hi = float(max(truths.max(), preds.max()))
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.3, lw=1, label="y = x")

    ax.set_xlabel("Ground truth affinity (kcal/mol)")
    ax.set_ylabel("Predicted affinity (kcal/mol)")
    ax.set_title(
        f"{title}\nPearson r = {r_p:.3f}   Spearman ρ = {r_s:.3f}   "
        f"MAE = {mae:.2f}   n = {len(rows)}"
    )
    ax.set_aspect("equal")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return {"pearson_r": float(r_p), "spearman_rho": float(r_s),
            "mae": mae, "rmse": rmse, "n": len(rows)}


def plot_residuals(rows: list[dict], out_path: Path, title: str) -> None:
    resid = np.array([r["pred"] - r["truth"] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.hist(resid, bins=30, color="tab:blue", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="k", lw=1, alpha=0.5)
    ax.axvline(float(resid.mean()), color="tab:red", lw=1.2, ls="--",
               label=f"mean = {resid.mean():.2f}")
    ax.set_xlabel("Prediction error (pred − truth, kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title} — residuals")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_by_confidence(rows: list[dict], out_path: Path, title: str) -> dict:
    by_conf = defaultdict(list)
    for r in rows:
        conf = r["confidence"] or "unspecified"
        by_conf[conf].append(abs(r["pred"] - r["truth"]))

    if len(by_conf) <= 1:
        return {k: {"n": len(v), "mean_abs_err": float(np.mean(v)) if v else None}
                for k, v in by_conf.items()}

    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    labels = [c for c in CONF_ORDER + ["unspecified"] if c in by_conf]
    data = [by_conf[l] for l in labels]
    colors = [CONF_COLOR.get(l, "tab:grey") for l in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "k", "markersize": 6})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("|prediction error| (kcal/mol)")
    ax.set_title(f"{title} — error by confidence")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return {k: {"n": len(v), "mean_abs_err": float(np.mean(v)) if v else None}
            for k, v in by_conf.items()}


def plot_case_sparklines(
    rows: list[dict],
    featurized_h5: Path | None,
    out_path: Path,
    title: str,
) -> None:
    if featurized_h5 is None or not featurized_h5.exists():
        return
    sorted_rows = sorted(rows, key=lambda r: abs(r["pred"] - r["truth"]))
    best = sorted_rows[0] if sorted_rows else None
    worst = sorted_rows[-1] if sorted_rows else None
    cases = [(best, "best (low error)"), (worst, "worst (high error)")]

    channel_names = [
        "min P–L dist", "mean contact dist", "contact count",
        "ligand RMSD", "radius of gyration", "buriedness",
    ]
    fig, axes = plt.subplots(len(cases), 6, figsize=(14, 4.5), dpi=140,
                             sharex=True)
    with h5py.File(featurized_h5, "r") as h5:
        h5_keys_lower = {k.lower(): k for k in h5.keys()}
        for row_idx, (case, label) in enumerate(cases):
            if case is None:
                continue
            key = h5_keys_lower.get(case["pdb_id"].lower())
            if key is None:
                continue
            feats = h5[key][:]
            for ch in range(6):
                ax = axes[row_idx, ch]
                ax.plot(feats[ch], lw=1.4, color="tab:blue")
                ax.set_title(channel_names[ch], fontsize=9)
                ax.grid(True, alpha=0.2)
                if ch == 0:
                    ax.set_ylabel(
                        f"{case['pdb_id']}\ntruth={case['truth']:.2f}\npred={case['pred']:.2f}",
                        fontsize=9,
                    )
            axes[row_idx, 0].annotate(label, xy=(-0.35, 0.5),
                                      xycoords="axes fraction",
                                      ha="right", va="center", fontsize=10,
                                      fontweight="bold")
    fig.suptitle(f"{title} — sample cases (6-channel trajectories)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def maybe_log_wandb(args: argparse.Namespace, plots: dict[str, Path], summary: dict) -> None:
    if not args.wandb_run_id:
        return
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping wandb upload")
        return
    project = args.wandb_project
    run = wandb.init(project=project, id=args.wandb_run_id, resume="must")
    log_payload = {f"vis/{name}": wandb.Image(str(path)) for name, path in plots.items()}
    log_payload.update({f"eval_summary/{k}": v for k, v in summary.items()
                        if isinstance(v, (int, float))})
    run.log(log_payload)
    run.finish()
    print(f"pushed {len(plots)} plots and {len(log_payload)} scalars to wandb run {args.wandb_run_id}")


def main(args: argparse.Namespace) -> int:
    rows = load_eval_csv(Path(args.csv))
    if len(rows) < 3:
        print(f"only {len(rows)} parseable rows in {args.csv} — need >= 3 to plot")
        return 1
    print(f"loaded {len(rows)} parseable predictions from {args.csv}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scatter_p = out_dir / "scatter.png"
    resid_p = out_dir / "residuals.png"
    by_conf_p = out_dir / "by_confidence.png"
    spark_p = out_dir / "sparklines.png"

    summary = plot_scatter(rows, scatter_p, args.title)
    print(f"  scatter -> {scatter_p}")
    plot_residuals(rows, resid_p, args.title)
    print(f"  residuals -> {resid_p}")
    by_conf_summary = plot_by_confidence(rows, by_conf_p, args.title)
    print(f"  by_confidence -> {by_conf_p}")
    plot_case_sparklines(rows, Path(args.featurized_h5) if args.featurized_h5 else None,
                         spark_p, args.title)
    if spark_p.exists():
        print(f"  sparklines -> {spark_p}")

    summary["per_confidence"] = by_conf_summary
    summary["title"] = args.title
    summary["csv"] = args.csv
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary -> {out_dir / 'summary.json'}")

    print(f"\nKey numbers:")
    print(f"  Pearson r  = {summary['pearson_r']:.4f}")
    print(f"  Spearman ρ = {summary['spearman_rho']:.4f}")
    print(f"  MAE        = {summary['mae']:.4f}  kcal/mol")
    print(f"  RMSE       = {summary['rmse']:.4f}  kcal/mol")
    print(f"  n          = {summary['n']}")

    plots = {"scatter": scatter_p, "residuals": resid_p, "by_confidence": by_conf_p}
    if spark_p.exists():
        plots["sparklines"] = spark_p
    maybe_log_wandb(args, plots, summary)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True,
                   help="path to eval CSV produced by scripts/eval_benchmark.py")
    p.add_argument("--title", required=True,
                   help='figure title, e.g. "MISATO test — final checkpoint"')
    p.add_argument("--output-dir", required=True,
                   help="directory to write PNGs + summary.json")
    p.add_argument("--featurized-h5", default="data/featurized.h5",
                   help="for the sample sparkline plot (best vs worst case)")
    p.add_argument("--wandb-run-id", default=None,
                   help="existing wandb run id (e.g. p06r0jb5) to attach plots to")
    p.add_argument("--wandb-project", default="tslm-md")
    sys.exit(main(p.parse_args()))
