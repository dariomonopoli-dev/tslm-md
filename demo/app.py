"""Streamlit live demo for TSLM-MD.

Layout:
  - Sidebar: paste/select PDB id, choose verifier threshold tau_high
  - Main: six per-frame feature sparklines from the trajectory
  - Below: predicted affinity + confidence, deterministic grounded rationale
  - Right: verifier comparison (predicted vs independent energy) + verdict badge

This skeleton runs with MOCK data so we can iterate on layout BEFORE training
finishes. Replace `mock_report()` with a real agent.run_agent() call at hour 18+.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from tslm_md.rationale import deterministic_rationale, channel_summary_dict, CHANNEL_NAMES

st.set_page_config(page_title="TSLM-MD — Binding-Affinity Agent", layout="wide")


def mock_features(seed: int = 0) -> torch.Tensor:
    """Build a plausible-shaped [6, 30] tensor so the demo renders before training is done."""
    rng = np.random.default_rng(seed)
    F = 30
    feats = np.zeros((6, F), dtype=np.float32)
    feats[0] = 3.5 + rng.normal(0, 0.2, F).cumsum() * 0.05   # min pocket distance
    feats[0] = np.clip(feats[0], 2.5, 5.0)
    feats[1] = feats[0] + 1.0 + rng.normal(0, 0.2, F)
    feats[2] = 10 + rng.integers(-3, 4, F).cumsum()
    feats[2] = np.clip(feats[2], 0, 25)
    feats[3] = np.abs(rng.normal(0, 0.3, F).cumsum() * 0.3)
    feats[4] = 4.0 + rng.normal(0, 0.1, F)
    feats[5] = 8 + rng.integers(-2, 3, F)
    return torch.from_numpy(feats)


def mock_report(pdb_id: str, tau_high: float) -> dict:
    feats = mock_features(seed=hash(pdb_id) % 2**32)
    rationale = deterministic_rationale(feats)
    summary = channel_summary_dict(feats)
    # Deterministic mock affinity from feature aggregates
    mock_affinity = -8.4 + (hash(pdb_id) % 70 - 35) / 20.0
    mock_independent = mock_affinity + (hash(pdb_id + "v") % 30 - 15) / 10.0
    disagreement = abs(mock_affinity - mock_independent) * 0.6   # mock z-score
    verdict = "INCONCLUSIVE" if disagreement > tau_high else "CONFIRMED"
    return {
        "pdb_id": pdb_id,
        "affinity": round(mock_affinity, 2),
        "confidence": "high" if disagreement < 0.5 else "medium",
        "independent_energy": round(mock_independent, 2),
        "disagreement_z": round(disagreement, 2),
        "rationale": rationale,
        "channel_summary": summary,
        "verdict": verdict,
        "verdict_reason": (
            f"|delta_z| = {disagreement:.2f} {'>' if verdict == 'INCONCLUSIVE' else '<='} tau_high = {tau_high}"
        ),
        "features": feats,
    }


# ---- UI

st.title("TSLM-MD — Binding-Affinity Agent")
st.caption(
    "Time-Series Language Model fine-tuned on molecular dynamics trajectories. "
    "Predicts binding affinity, generates grounded rationale, verifies against independent physics, abstains on disagreement."
)

with st.sidebar:
    st.header("Input")
    pdb_id = st.text_input("PDB id", value="1A4K", help="any 4-char PDB id present in MISATO")
    tau_high = st.slider("Abstention threshold tau_high (|delta_z|)", 0.5, 3.0, 1.5, 0.1)
    if st.button("Analyse", type="primary"):
        st.session_state["report"] = mock_report(pdb_id, tau_high)
    st.divider()
    st.markdown(
        "**Demo mode:** outputs are mock data so the layout works before training finishes. "
        "Real model wired in at hour 18+."
    )

report = st.session_state.get("report")
if not report:
    st.info("Paste a PDB id and click **Analyse** to begin.")
    st.stop()

# ---- top row: feature sparklines

st.subheader(f"Trajectory features — PDB id {report['pdb_id']}")
cols = st.columns(6)
for i, name in enumerate(CHANNEL_NAMES):
    series = report["features"][i].numpy()
    cols[i].metric(label=name, value=f"{series[-1]:.2f}", delta=f"{series[-1] - series[0]:+.2f}")
    cols[i].line_chart(series, height=120)

# ---- middle: prediction + verdict

left, right = st.columns([2, 1])
with left:
    st.subheader("Prediction")
    st.metric("Predicted affinity (kcal/mol)", f"{report['affinity']}")
    st.metric("Model confidence", report["confidence"] or "n/a")
    st.subheader("Grounded rationale (deterministic, from trajectory data)")
    st.write(report["rationale"])

with right:
    st.subheader("Verdict")
    if report["verdict"] == "CONFIRMED":
        st.success(f"✅ CONFIRMED")
    elif report["verdict"] == "INCONCLUSIVE":
        st.warning(f"⚠ INCONCLUSIVE")
    else:
        st.error(f"❌ {report['verdict']}")
    st.caption(report["verdict_reason"])
    st.divider()
    st.subheader("Independent physics verifier")
    st.metric("Independent energy", f"{report['independent_energy']}")
    st.metric("|delta_z|", f"{report['disagreement_z']}")
