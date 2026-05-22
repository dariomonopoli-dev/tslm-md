"""Post-hoc verifier for OpenTSLM-SP rationales.

============================  Where the data comes from  ============================

Two inputs:
  1. The rationale text - a string. At training time we use the templated rationales
     in `preprocessed/samples_{split}.jsonl` (rendered programmatically from facts).
     At inference time the rationale comes from `model.generate(batch)`.
  2. The ground-truth facts dict for that PDB, taken from
     `preprocessed/samples_{split}.jsonl` -> sample["facts"].

================================  How verification works  ============================

Five claim types per PROJECT_BRIEF §7.2. Each claim type:
  - Has a regex that pulls candidate phrases out of the rationale
  - Has a verifier that compares the claimed numeric value to the corresponding
    field of facts["summary"] or facts["events"]

Each extracted claim resolves to one of:
  verified      |x_claim - x_fact| within tolerance
  contradicted  |x_claim - x_fact| outside tolerance
  unverifiable  regex matched a phrase but no corresponding fact exists

Anything in the rationale that does NOT match any claim regex is silently
ignored (it's outside the closed claim vocabulary per §7.2).

================================  How to read the output  ============================

Reported as a per-rationale breakdown plus an aggregate "% verified" over
all extracted claims. The headline metric in the writeup is

    % verified  =  verified / (verified + contradicted)

(unverifiable claims are excluded - we can't grade them with the 4 input channels).

================================  CLI usage  ============================

Self-test on the templated training rationales (should be ~100% verified):
    python verify_rationale.py --self-test

Verify a JSONL of model-generated rationales (one per line: {"pdb_id": ..., "rationale": ...}):
    python verify_rationale.py --predictions runs/v1a_*/eval_test.jsonl --split test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Tolerances per claim. Looser than the templater's precision so light paraphrases pass.
TOL = {
    "energy_mean": 1.0,           # kcal/mol
    "energy_range": 2.0,
    "rmsd_mean": 0.10,            # A
    "rmsd_max": 0.10,
    "bsasa_mean": 10.0,           # A^2
    "pocket_residence_frac": 0.05,  # absolute fraction
    "energy_spike": 5.0,
    "ligand_drift_rmsd": 0.5,
    "bsasa_drop": 30.0,
}


@dataclass
class Verdict:
    claim_type: str
    field: str
    claimed: float
    actual: Optional[float]
    status: str  # verified | contradicted | unverifiable
    detail: str = ""


@dataclass
class Report:
    pdb_id: str
    verdicts: list[Verdict] = field(default_factory=list)

    def counts(self) -> dict:
        c = {"verified": 0, "contradicted": 0, "unverifiable": 0}
        for v in self.verdicts:
            c[v.status] += 1
        return c


# ---------- claim regexes ----------
# These match the templater in preprocess_misato.py:render_rationale exactly,
# but tolerate small numeric / wording variation.

R_ENERGY_MEAN = re.compile(
    r"[Mm]ean (?:interaction )?energy (?:was|of|is)\s+(-?\d+(?:\.\d+)?)\s*kcal/mol", re.I)
R_ENERGY_RANGE = re.compile(
    r"swing of\s+(\d+(?:\.\d+)?)\s*kcal/mol", re.I)
R_RMSD_MEAN = re.compile(
    r"[Ll]igand RMSD\s+(?:averaged|of|was)\s+(\d+(?:\.\d+)?)\s*A", re.I)
R_RMSD_MAX = re.compile(
    r"max(?:imum)?\s+(\d+(?:\.\d+)?)(?:\s*A)?", re.I)
R_BSASA_MEAN = re.compile(
    r"[Bb]uried SASA\s+(?:averaged|was|of)\s+(\d+(?:\.\d+)?)\s*A\^?2", re.I)
R_POCKET_PCT = re.compile(
    r"stayed within\s+\d+(?:\.\d+)?\s*A.*?for\s+(\d+(?:\.\d+)?)\s*%", re.I | re.S)
R_ENERGY_SPIKE = re.compile(
    r"[Ee]nergy jumps from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)\s*kcal/mol\s+at frame\s+(\d+)",
    re.I)
R_LIGAND_DRIFT = re.compile(
    r"ligand drifts by\s+(\d+(?:\.\d+)?)\s*A", re.I)
R_BSASA_DROP = re.compile(
    r"[Bb]uried SASA drops from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*A\^?2", re.I)


def _check(claimed: float, actual: Optional[float], tol: float, claim_type: str, field_name: str) -> Verdict:
    if actual is None:
        return Verdict(claim_type, field_name, claimed, None, "unverifiable",
                       "no fact for this field")
    err = abs(claimed - actual)
    status = "verified" if err <= tol else "contradicted"
    return Verdict(claim_type, field_name, claimed, actual, status,
                   f"|claim - fact| = {err:.3f}, tol = {tol}")


def verify_rationale(rationale: str, facts: dict) -> Report:
    """Verify every numeric claim in `rationale` against `facts`."""
    summary = facts.get("summary", {})
    events = facts.get("events", [])
    pdb_id = facts.get("pdb_id", "?")
    rep = Report(pdb_id=pdb_id)

    if m := R_ENERGY_MEAN.search(rationale):
        rep.verdicts.append(_check(float(m.group(1)), summary.get("energy_mean"),
                                   TOL["energy_mean"], "energy_trend", "energy_mean"))
    if m := R_ENERGY_RANGE.search(rationale):
        rep.verdicts.append(_check(float(m.group(1)), summary.get("energy_range"),
                                   TOL["energy_range"], "energy_trend", "energy_range"))
    if m := R_RMSD_MEAN.search(rationale):
        rep.verdicts.append(_check(float(m.group(1)), summary.get("rmsd_mean"),
                                   TOL["rmsd_mean"], "rmsd_stability", "rmsd_mean"))
    # rmsd_max is the trickiest pattern (`max X` can match many contexts) — restrict
    # to phrases inside an RMSD parenthetical: "(max 2.38)".
    for m in re.finditer(r"RMSD.{0,40}?\(max\s+(\d+(?:\.\d+)?)\)", rationale, re.I):
        rep.verdicts.append(_check(float(m.group(1)), summary.get("rmsd_max"),
                                   TOL["rmsd_max"], "rmsd_stability", "rmsd_max"))
    if m := R_BSASA_MEAN.search(rationale):
        rep.verdicts.append(_check(float(m.group(1)), summary.get("bsasa_mean"),
                                   TOL["bsasa_mean"], "contact_persistence", "bsasa_mean"))
    if m := R_POCKET_PCT.search(rationale):
        claimed_frac = float(m.group(1)) / 100.0
        rep.verdicts.append(_check(claimed_frac, summary.get("pocket_residence_fraction"),
                                   TOL["pocket_residence_frac"], "pocket_residence",
                                   "pocket_residence_fraction"))

    # Event claims: match against the first event of each type
    by_type = {}
    for e in events:
        by_type.setdefault(e["type"], e)

    if m := R_ENERGY_SPIKE.search(rationale):
        e = by_type.get("energy_spike")
        if e is None:
            rep.verdicts.append(Verdict("energy_trend", "energy_spike_to", float(m.group(2)),
                                        None, "unverifiable", "no energy_spike event"))
        else:
            rep.verdicts.append(_check(float(m.group(1)), e["from"], TOL["energy_spike"],
                                       "energy_trend", "energy_spike_from"))
            rep.verdicts.append(_check(float(m.group(2)), e["to"], TOL["energy_spike"],
                                       "energy_trend", "energy_spike_to"))

    if m := R_LIGAND_DRIFT.search(rationale):
        e = by_type.get("ligand_drift")
        if e is None:
            rep.verdicts.append(Verdict("rmsd_stability", "ligand_drift_rmsd",
                                        float(m.group(1)), None, "unverifiable",
                                        "no ligand_drift event"))
        else:
            rep.verdicts.append(_check(float(m.group(1)), e["delta_rmsd"],
                                       TOL["ligand_drift_rmsd"], "rmsd_stability",
                                       "ligand_drift_rmsd"))

    if m := R_BSASA_DROP.search(rationale):
        e = by_type.get("contact_drop")
        if e is None:
            rep.verdicts.append(Verdict("contact_persistence", "bsasa_drop_to",
                                        float(m.group(2)), None, "unverifiable",
                                        "no contact_drop event"))
        else:
            rep.verdicts.append(_check(float(m.group(1)), e["from"], TOL["bsasa_drop"],
                                       "contact_persistence", "bsasa_drop_from"))
            rep.verdicts.append(_check(float(m.group(2)), e["to"], TOL["bsasa_drop"],
                                       "contact_persistence", "bsasa_drop_to"))

    return rep


def aggregate(reports: list[Report]) -> dict:
    totals = {"verified": 0, "contradicted": 0, "unverifiable": 0, "n_rationales": 0, "n_claims": 0}
    for r in reports:
        c = r.counts()
        for k, v in c.items():
            totals[k] += v
        totals["n_claims"] += sum(c.values())
        totals["n_rationales"] += 1
    grounded = totals["verified"] + totals["contradicted"]
    totals["pct_verified_of_grounded"] = (100.0 * totals["verified"] / grounded) if grounded else float("nan")
    totals["pct_claims_unverifiable"] = (100.0 * totals["unverifiable"] / max(1, totals["n_claims"]))
    totals["mean_claims_per_rationale"] = totals["n_claims"] / max(1, totals["n_rationales"])
    return totals


def self_test(data_dir: Path, split: str = "train", limit: int = 200) -> None:
    """Run the verifier on the templated training rationales themselves.

    Since those are rendered FROM the facts, % verified should be ~100%.
    A perturbation pass injects bogus numbers and confirms the verifier catches them.
    """
    samples_path = data_dir / f"samples_{split}.jsonl"
    print(f"self-test: reading first {limit} from {samples_path}")
    reports = []
    perturbed_reports = []
    with samples_path.open() as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            s = json.loads(line)
            facts = s["facts"]
            facts.setdefault("pdb_id", s["pdb_id"])
            rep = verify_rationale(s["rationale"], facts)
            reports.append(rep)

            # Perturbed copy: replace every number in the rationale with itself + 100.
            perturbed = re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group(0)) + 100:.1f}", s["rationale"])
            perturbed_reports.append(verify_rationale(perturbed, facts))

    agg_clean = aggregate(reports)
    agg_perturb = aggregate(perturbed_reports)
    print()
    print("ORIGINAL TEMPLATED RATIONALES (expect ~100% verified):")
    print(json.dumps(agg_clean, indent=2))
    print()
    print("PERTURBED RATIONALES (every number + 100; expect ~0% verified):")
    print(json.dumps(agg_perturb, indent=2))


def verify_file(predictions_path: Path, data_dir: Path, split: str) -> dict:
    """Verify a JSONL of model-generated rationales against the ground-truth facts."""
    # Build pdb -> facts lookup from the samples file.
    samples_path = data_dir / f"samples_{split}.jsonl"
    fact_lookup = {}
    with samples_path.open() as f:
        for line in f:
            s = json.loads(line)
            fact_lookup[s["pdb_id"]] = s["facts"]

    reports: list[Report] = []
    with predictions_path.open() as f:
        for line in f:
            p = json.loads(line)
            facts = fact_lookup.get(p["pdb_id"])
            if facts is None:
                print(f"warn: no facts for {p['pdb_id']}", file=sys.stderr)
                continue
            facts.setdefault("pdb_id", p["pdb_id"])
            reports.append(verify_rationale(p["rationale"], facts))
    agg = aggregate(reports)
    print(json.dumps(agg, indent=2))
    return {"aggregate": agg, "per_rationale": [{"pdb_id": r.pdb_id, **r.counts()} for r in reports]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--self-test", action="store_true",
                    help="Verify the templated rationales + perturbed copies.")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="JSONL of model-generated rationales: {pdb_id, rationale}.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    if args.self_test:
        self_test(args.data_dir, split=args.split, limit=args.limit)
    elif args.predictions:
        out = verify_file(args.predictions, args.data_dir, args.split)
        if args.out:
            args.out.write_text(json.dumps(out, indent=2))
    else:
        ap.error("pass --self-test or --predictions <file>")


if __name__ == "__main__":
    main()
