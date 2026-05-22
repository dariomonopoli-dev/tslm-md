"""Generate data/failure_modes_{variant}.json from real tunnel predictions.

Picks N sample PDBs from the local test split, calls tunnel /predict for
each, looks up ground truth, ranks by |Δ|, builds reasoning text from
tunnel verdict_reason + worst-channel observation. Writes JSON the
frontend reads at /failure_modes.

Designed to be re-runnable when the tunnel checkpoint changes — the
output is committed to the repo so the demo works out of the box without
any precompute.

Usage:
    export TUNNEL_URL=https://your-tunnel-url
    python scripts/generate_failure_modes.py --variant v1b --n-sample 30 --n-take 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Allow `python scripts/generate_failure_modes.py` from inference-service/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference


_RT_LN10_KCAL = 1.3633


def _tunnel_predict(tunnel_url: str, pdb_id: str) -> dict | None:
    body = json.dumps({"pdb_id": pdb_id}).encode()
    req = urllib.request.Request(
        f"{tunnel_url.rstrip('/')}/predict",
        data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  {pdb_id}: HTTP {e.code}")
        return None
    except Exception as e:
        print(f"  {pdb_id}: {e}")
        return None


def _worst_channel_note(channel_summary: dict) -> str:
    """Pick the channel with the largest start→end change as a key talking point."""
    if not channel_summary:
        return ""
    candidates = []
    for name, stats in channel_summary.items():
        try:
            start = float(stats["start"])
            end = float(stats["end"])
            delta = abs(end - start)
            trend = stats.get("trend", "")
            mean = float(stats.get("mean", 0))
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((delta, name, start, end, trend, mean))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    _, name, start, end, trend, mean = candidates[0]
    return f"{name} {trend} ({start:.2f} → {end:.2f}, mean {mean:.2f})"


def _bucket_pattern(reason: str) -> str:
    r = (reason or "").lower()
    # Tunnel-verdict patterns (most common today)
    if "|z(model) - z(physics)|" in r or "disagreement" in r:
        return "Model–physics disagreement"
    if "low confidence" in r or "inconclusive" in r:
        return "Model self-flagged uncertainty"
    if "agreement within threshold" in r and "experimental" in r:
        return "Confident model, experimental mismatch"
    # Trajectory/channel-driven patterns
    if "ligand efficien" in r or "implausible" in r:
        return "Implausible ligand efficiency"
    if ("rmsd" in r and "increasing" in r) or "unbinding" in r or "drift" in r:
        return "Pose drift / unbinding"
    if "outlier" in r or "clash" in r or "broken" in r:
        return "Single-frame outlier dominates"
    if "bsasa" in r or "buried" in r and ("decreasing" in r or "drop" in r):
        return "Contact loss (bSASA decreasing)"
    if "literature" in r or "contradict" in r:
        return "Literature contradicts binding mode"
    return "Other"


def generate(variant: str, n_sample: int, n_take: int,
             tunnel_url: str, out_path: Path) -> None:
    # Load ground-truth + tunnel health (reuses inference.py logic).
    inference._load_ground_truth()

    candidates = sorted(inference._GROUND_TRUTH.keys())
    random.seed(42)
    sample = random.sample(candidates, min(n_sample, len(candidates)))
    print(f"[generate] sampling {len(sample)} PDBs through tunnel {tunnel_url}")

    rows: list[dict] = []
    for i, pid in enumerate(sample, 1):
        data = _tunnel_predict(tunnel_url, pid)
        if not data:
            continue
        affinity = float(data.get("affinity", 0.0))
        pred_pk = -affinity / _RT_LN10_KCAL
        truth = float(inference._GROUND_TRUTH[pid])
        delta = abs(pred_pk - truth)
        verdict = data.get("verdict", "?")
        confidence = data.get("confidence", "?")
        verdict_reason = (data.get("verdict_reason") or "").strip()
        worst = _worst_channel_note(data.get("channel_summary") or {})

        reason_parts = []
        if verdict_reason:
            reason_parts.append(verdict_reason)
        reason_parts.append(
            f"predicted pK {pred_pk:.2f} vs experimental {truth:.2f} (|Δ|={delta:.2f})"
        )
        if worst:
            reason_parts.append(f"worst trend: {worst}")
        reason = "; ".join(reason_parts)

        # Convert verdict to recommendation pill enum the UI expects.
        if verdict == "CONFIRMED" and delta < 1.0:
            agent = "trust"
        elif verdict in ("INCONCLUSIVE", "?") or delta >= 1.5:
            agent = "review" if delta < 2.5 else "discard"
        else:
            agent = "review"

        rows.append({
            "pdb": pid,
            "model": round(pred_pk, 2),
            "vina": round(float(data.get("independent_energy", 0.0)) / -_RT_LN10_KCAL, 2),
            "mmgbsa": round(truth, 2),     # using ground truth as the "physics anchor" slot
            "agent": agent,
            "reason": reason,
            "_delta": delta,                # for sorting
            "_verdict": verdict,
            "_confidence": confidence,
        })
        if i % 5 == 0:
            print(f"  {i}/{len(sample)} … {len(rows)} predictions collected")

    rows.sort(key=lambda r: r["_delta"], reverse=True)
    top = rows[:n_take]

    # Strip the internal sort keys before serializing.
    for r in top:
        r.pop("_delta", None)
        r.pop("_verdict", None)
        r.pop("_confidence", None)

    # Aggregate failure pattern buckets.
    buckets: dict[str, list[str]] = {}
    for r in top:
        bucket = _bucket_pattern(r["reason"])
        buckets.setdefault(bucket, []).append(r["pdb"])
    patterns = [
        {"cluster": k, "count": len(v), "systems": ", ".join(v)}
        for k, v in buckets.items()
    ]

    payload = {
        "variant": variant,
        "rows": top,
        "patterns": patterns,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n[generate] wrote {len(top)} rows + {len(patterns)} pattern buckets → {out_path}")


def main() -> None:
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["v1a", "v1b", "all"], default="all")
    p.add_argument("--n-sample", type=int, default=30,
                   help="How many PDBs to predict through the tunnel (default 30)")
    p.add_argument("--n-take", type=int, default=10,
                   help="Top-N by |Δ| to keep in the output (default 10)")
    p.add_argument("--tunnel-url", default=os.getenv("TUNNEL_URL", ""))
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    if not args.tunnel_url:
        raise SystemExit("set TUNNEL_URL env var or pass --tunnel-url")

    variants = ["v1a", "v1b"] if args.variant == "all" else [args.variant]
    for v in variants:
        out = args.out_dir / f"failure_modes_{v}.json"
        print(f"\n== {v} ==")
        generate(v, args.n_sample, args.n_take, args.tunnel_url, out)


if __name__ == "__main__":
    main()
