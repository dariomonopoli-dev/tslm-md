"""Streamlit live demo for TSLM-MD.

Stub: implement during hour 18-22.

Layout:
  - Sidebar: paste PDB id, press [Analyse]
  - Main: 6 feature sparklines from the trajectory
  - Below: streaming LM rationale, parsed affinity + confidence
  - Right column: verifier verdict (CONFIRMED / INCONCLUSIVE) with reason
  - Bottom: comparison table (predicted vs independent vs ground-truth)
"""

import streamlit as st

st.set_page_config(page_title="TSLM-MD", layout="wide")
st.title("TSLM-MD — Binding-Affinity Agent")
st.write("Paste a PDB id and watch the agent reason over its MD trajectory.")
st.info("Demo stub — implement during hour 18-22.")
