# MD-Trajectory Binding Affinity with Grounded Rationales

**Applying OpenTSLM-SoftPrompt to a new modality: protein-ligand molecular dynamics.**

One-line pitch: most binding-affinity predictors look at a static snapshot of a drug sitting in a protein. We read the **movie** — the molecular-dynamics trajectory of how the drug wiggles in the pocket over time — and output (1) a predicted binding affinity that beats a docking baseline, and (2) a natural-language rationale explaining why ("binding destabilizes after ~3 ns as the ligand drifts out of contact"). A post-hoc verifier checks that every claim in the rationale is consistent with the underlying energy/structure data, so the explanation is grounded, not hallucinated. The artifact is **trained weights**, not a prompt wrapper.

---

## 1. Goal

Take the published OpenTSLM-SoftPrompt method (Stanford BDHG et al., *arXiv 2510.02410*, ICML 2026) and apply it to a modality it has not been tried on — protein-ligand MD trajectories — using the MISATO dataset for inputs and the `misato-affinity` companion dataset for labels. Output is a fine-tuned checkpoint that:

1. Regresses binding affinity (as a generated string `Answer: <pK>`).
2. Generates a natural-language rationale that describes the trajectory's dynamics.
3. Has its rationales verified post-hoc against the trajectory's own per-frame quantities.

We train **two variants** and report both:

- **v1a — Faithful.** Pure OpenTSLM-SP exactly as published. Affinity is generated as a string token (`Answer: 6.42`) and parsed at inference time. This is the honest "applies the published method to a new modality" claim.
- **v1b — Hybrid.** Adds a scalar regression head on top of the LLM's final input-position hidden state. Joint loss `L = L_LM + λ · MSE(pK_pred, pK_true)`. This is a small, well-scoped extension of the method (~40 lines).

The ablation v1a vs v1b answers a question reviewers will ask: *does extending the method actually help?* If v1b wins by a wide margin, that's a finding. If v1a is close, the published method transfers cleanly to MD and no extension is needed. Either outcome is publishable.

The contribution is **modality + grounded rationales + an honest ablation on the regression-head extension** — not a new training algorithm.

---

## 2. Team and constraints

| Constraint | Value |
|---|---|
| Wall-clock budget | **20 hours** |
| Team size | **4 people** (≈ 80 person-hours) |
| Compute | **AWS SageMaker** (recommended: `ml.g5.xlarge` spot, ~$0.40/hr) |
| Cost ceiling | ~$30–50 of compute |
| Base model | `OpenTSLM/llama-3.2-1b-tsqa-sp` (1 B params + LoRA) |
| Deliverable | Trained checkpoint + demo notebook + 1-page result writeup |

This is a **demo-grade artifact**, not a paper-grade benchmark. Be honest about that with stakeholders.

---

## 3. Method in one picture

```
MD trajectory (100 frames per system)
        │
        ▼  load 4 precomputed per-frame channels from MISATO HDF5:
            • frames_rmsd_ligand        (100,)
            • frames_interaction_energy (100,)
            • frames_distance           (100,)
            • frames_bSASA              (100,)
        │
        ▼  4 univariate time-series, normalized per channel
        │
   OpenTSLM-SoftPrompt
   (Llama-3.2-1B, frozen + LoRA)
        │
        ▼  generates a single string:
        "<rationale paragraph> Answer: 6.42"
        │
        ▼  parse "Answer: X.XX" → affinity prediction
        │
   ┌────┴───────────────┐
   ▼                    ▼
 affinity metric    rationale verifier
 (Pearson R)        (claim extractor → fact lookup → % verified)
```

No new architecture. The "new modality" claim is honest because OpenTSLM-SP has only been published on ECG/EEG/accel/M4 — never on MD.

---

## 4. What is on disk right now

All paths under `/home/mxlk/Documents/AIproject /` (trailing space in folder name — quote it in shell).

### 4.1 MISATO dataset (`misato-dataset-master/`)

Contains the codebase, tiny demo data, and full splits. The **133 GB full MD HDF5 is downloading separately** to S3.

| File | Size | Content |
|---|---|---|
| `data/MD/h5_files/tiny_md.hdf5` | 93 MB | 20 systems, full schema (works as a smoke test now) |
| `data/MD/h5_files/tiny_md_out.hdf5` | 48 MB | 20 systems, with adaptability outputs |
| `data/MD/splits/train_MD.txt` | 68 KB | 13,765 PDB IDs |
| `data/MD/splits/val_MD.txt` | 7.8 KB | 1,595 PDB IDs |
| `data/MD/splits/test_MD.txt` | 7.9 KB | 1,612 PDB IDs |
| `data/MD/restart/11gs/` | 10 MB | AMBER restart + topology (only one system) |
| `data/peptides.txt` | 8 KB | 1,430 PDB IDs flagged as peptides |
| `src/data/...` | — | PyTorch dataloaders + processing scripts |

**Per-system MD schema** (verified empirically on tiny):

```
<PDB>/
  trajectory_coordinates          (100, N_atoms, 3)   float64
  atoms_element / atoms_residue / atoms_type / atoms_number  (N_atoms,)
  molecules_begin_atom_index      (3,)
  frames_rmsd_ligand              (100,)              ← TS channel 1
  frames_interaction_energy       (100,)              ← TS channel 2 (strongest affinity signal)
  frames_distance                 (100,)              ← TS channel 3
  frames_bSASA                    (100,)              ← TS channel 4
```

### 4.2 QM dataset (`QM (1).hdf5`, top-level)

19,413 ligands. Per-ligand: 28 per-atom features (xyz, hybridization, 5 partial-charge schemes, polarization, electrophilicity/nucleophilicity), bonds list, 7 scalar molecular properties (electron affinity, hardness, etc.). **Not needed for v1**; useful for an ablation that adds QM features.

### 4.3 Affinity labels (`misato-affinity/`)

| File | Size | Content |
|---|---|---|
| `data/affinity_data.csv` | 1.2 MB | **19,443 rows**, PDB → Kd/Ki/IC50 (in nM) + ligand/uniprot/protein |
| `data/affinity_data.h5` | 65 MB | Structural features (used by their GCN baseline) |
| `data/train_pairs.pickle` | 154 KB | 11,076 PDB pairs (their split scheme) |
| `data/val_pairs.pickle` | 17 KB | 1,199 pairs |
| `data/test_pairs.pickle` | 1.9 KB | 163 pairs |
| `configs/best_ckpts/` | — | Their pretrained GCN checkpoints (the baseline to beat) |

**pK formula:**
```
pK = 9 - log10(value_in_nM)
```
Priority order when multiple columns are filled: **Kd > Ki > IC50** (only the first non-zero is used).

### 4.4 OpenTSLM source (`OpenTSLM/`)

| Path | Role |
|---|---|
| `src/opentslm/model/llm/OpenTSLMSP.py` | The SP model (frozen LLM + internal TS encoder + LoRA) |
| `src/opentslm/model/encoder/TransformerCNNEncoder.py` | The TS encoder (Conv1d patch + 6-layer Transformer) |
| `src/opentslm/time_series_datasets/QADataset.py` | Abstract task base |
| `src/opentslm/time_series_datasets/har_cot/HARCoTQADataset.py` | **Template to copy** — classification + CoT rationale |
| `src/opentslm/prompt/text_time_series_prompt.py` | Per-channel container (hard constraint: 1-D per channel) |
| `curriculum_learning.py` | Trainer entrypoint, has 5 stages, we add a 6th |
| `pyproject.toml` | Requires Python ≥ 3.12, torch ≥ 2.9, transformers ≥ 4.57, peft ≥ 0.18 |

---

## 5. Empirical findings that shape the plan

We checked the data on disk before committing. Two of three early worries were wrong; one is partly mitigated.

### 5.1 Trajectory variance — substantial (worry refuted)

Tiny MD sample (20 systems, 100 frames each):

| Quantity | Median range per system | p90 | Max |
|---|---|---|---|
| Ligand RMSD (Å) | 1.83 | 3.80 | 6.12 |
| Interaction energy (kcal/mol) | 22.0 | 62.1 | 142.9 |
| Ligand-protein distance (Å) | 1.95 | 2.94 | 6.54 |
| Buried SASA (Å²) | 184.8 | 282.9 | 461.3 |

All 20 systems show meaningful per-frame variation. Several show dramatic motion (`1A09` swings 88 kcal/mol; `184L`, `1A08` ligands move >10 Å RMSD). The "movie" framing is supported by the data, not just rhetoric.

### 5.2 Affinity-label coverage — essentially total (worry refuted)

| Split | Overlap with `affinity_data.csv` |
|---|---|
| train_MD | 13,758 / 13,765 (**99.9%**) |
| val_MD | 1,595 / 1,595 (**100%**) |
| test_MD | 1,612 / 1,612 (**100%**) |
| **total** | **16,965 / 16,972 (100.0%)** |

pK distribution: min 0.40, p10 3.90, median 6.41, p90 8.66, max 15.22, mean 6.36, std 1.86. Looks like real PDBbind-style affinity data.

### 5.3 10 ns timescale — still a real limit (acknowledged)

10 ns cannot sample microsecond binding/unbinding equilibria, so a 10 ns sample cannot fully determine an experimental ΔG. There is a hard floor on accuracy from this. However, the per-frame interaction-energy variation already spans 20–60 kcal/mol within 10 ns, so the input does carry ensemble information beyond any single snapshot. We acknowledge the limit explicitly in any writeup.

---

## 6. Architecture — how the new task plugs into OpenTSLM

OpenTSLM-SP supervises only on the **answer string**. Multivariate inputs are passed as **multiple univariate `TextTimeSeriesPrompt`s**, each with its own text label (HAR uses 3 channels for x/y/z accel; we use 4). The TS encoder is internal; the LLM is frozen with LoRA on q/k/v/o + MLP projections.

### 6.1 Files to add (two of them)

```
src/opentslm/time_series_datasets/misato/
  __init__.py
  misato_loader.py            ← new, ~80 lines: H5 + CSV → train/val/test datasets
  MISATOMDQADataset.py        ← new, ~150 lines: QADataset subclass

curriculum_learning.py
  + import MISATOMDQADataset
  + def stage6_misato(...)
  + register "stage6_misato" in CURRICULUM_STAGES
```

### 6.2 Channels (input)

| Index | Channel | Source | Why it matters |
|---|---|---|---|
| 0 | Ligand RMSD vs frame 0 | `frames_rmsd_ligand` | Pose stability |
| 1 | Interaction energy | `frames_interaction_energy` | Strongest direct affinity signal |
| 2 | Ligand-protein distance | `frames_distance` | Engagement / drift |
| 3 | Buried SASA | `frames_bSASA` | Contact surface persistence |

Each channel is normalized per-system (mean/std) before going into the encoder, following HAR-CoT's convention. The text label per channel includes the original mean/std so the model can ground qualitative claims in absolute values.

### 6.3 Target (output)

A single generated string of the form:

> *"During the trajectory the ligand-protein interaction energy averages −37.2 kcal/mol and stabilises after frame 20. Ligand RMSD remains under 2.5 Å with buried SASA holding above 500 Å². The pose is stable. Answer: 6.42"*

Cross-entropy loss is on the entire answer string. The `Answer: <float>` suffix is parsed at evaluation time to give the affinity number.

### 6.4 v1a — faithful path (no architectural change)

- Multivariate-as-multiple-univariate: same as HAR-CoT, SleepEDF-CoT, ECG-CoT
- Pure LM loss, no extra regression head: same as every other OpenTSLM-SP stage
- LoRA on the LLM, full updates on encoder + projector: same as stages 3–5
- Affinity-as-token: same trick as classification stages, which predict the class as a token

No architectural changes. The new modality is the only delta from the published method. This is what we report as the "applies OpenTSLM-SP to a new modality" result.

### 6.5 v1b — hybrid path (regression head extension)

**Acknowledged as an extension of the published method**, not just an application. Trained alongside v1a so the ablation is honest.

**Architectural delta:** add a 2-layer MLP regression head on the LLM's last input-position hidden state. Joint loss `L = L_LM + λ · MSE(pK_pred, pK_true)` with `λ ≈ 0.5` (tune in v1b's first hour of training).

**Code change** (in `OpenTSLMSP.py`, ~40 lines total):

1. `__init__`: set `self.regression_enabled = False` and `self.regression_head = None`.
2. New method `enable_regression(weight=0.5)`: instantiates `nn.Sequential(Linear → GELU → Dropout → Linear)` head (~0.5 M params for the 2048-dim hidden state of Llama-3.2-1B).
3. `compute_loss`: if `self.regression_enabled`, set `output_hidden_states=True` on the LLM forward, pool the hidden state at the last non-pad input position per sample, run the head, compute MSE against `batch["pK"]`, and add it to the LM loss.
4. `store_to_file` / `load_from_file`: round-trip `regression_head.state_dict()`.
5. New method `generate_with_pK(batch)`: returns both the generated rationale string and the regression-head scalar prediction.

**Dataset delta** (in `MISATOMDQADataset._format_sample`):

```python
sample = super()._format_sample(row)
sample["pK"] = float(row["pK"])
return sample
```

That's the whole change.

**Pooling rationale.** We pool at the **last non-pad input position** — the model's "ready to answer" summary token after consuming all four TS channels and the prompt. Mean-pooling is the fallback if the last-token pool underfits; it's a one-line swap.

**Honest framing for the writeup:**

> "We trained two variants of OpenTSLM-SP on MISATO MD trajectories: a faithful variant (v1a) that reproduces the published method on a new modality with affinity emitted as a generated token, and a hybrid variant (v1b) that adds a scalar regression head on the model's final input-position hidden state. v1b extends the published architecture. We report both."

---

## 7. Labeling strategy — per-frame facts

The model is supervised on a single answer string per system. But to make the rationale **grounded**, we additionally compute a per-frame fact dict for each system. The fact dict serves three roles:

| Role | When | How |
|---|---|---|
| Source for the rationale text | Training | Templated from `facts` → answer string |
| Ground-truth for the verifier | Evaluation | Generated rationale → claim extractor → lookup in `facts` |
| (Optional) auxiliary supervision | v2 only | Per-frame head on the encoder output |

### 7.1 Fact dict schema

```python
facts = {
    "pdb_id": "1A1B",
    "summary": {
        "rmsd_mean": float, "rmsd_max": float,
        "energy_mean": float, "energy_range": float,
        "contacts_persistent": bool,
        "ligand_drift": bool,
    },
    "events": [
        {"type": "energy_spike",   "frame": 47, "from": -50.2, "to": -22.1},
        {"type": "ligand_drift",   "frame_range": [70, 99], "delta_rmsd": 3.6},
        {"type": "contact_drop",   "frame_range": [70, 80], "from": 540.0, "to": 320.0},
    ],
}
```

### 7.2 Closed claim vocabulary (frozen at hour 0)

Five claim types. Every claim type must be machine-checkable from the four input channels alone (no topology required).

| Claim type | Verified from |
|---|---|
| `rmsd_stability` | mean ± std of channel 0 |
| `pocket_residence` | fraction of frames with channel 2 below a threshold |
| `contact_persistence` | mean and slope of channel 3 (bSASA) |
| `energy_trend` | sign and magnitude of slope of channel 1 |
| `flexibility` | std of channel 0 |

Five is enough for credible rationales and small enough that the verifier is regex-grade. If a generated rationale mentions something **outside** this vocabulary, the verifier marks it `unverifiable` and we discount it.

### 7.3 Rationale template (programmatic)

```python
def render(facts):
    s = facts["summary"]
    text = (
        f"Mean interaction energy was {s['energy_mean']:.1f} kcal/mol "
        f"with a swing of {s['energy_range']:.1f} kcal/mol. "
        f"Ligand RMSD averaged {s['rmsd_mean']:.2f} Å (max {s['rmsd_max']:.2f}). "
    )
    for e in facts["events"]:
        if e["type"] == "ligand_drift":
            text += f"Between frames {e['frame_range'][0]} and {e['frame_range'][1]} the ligand drifts by {e['delta_rmsd']:.1f} Å. "
        ...
    text += f"Answer: {facts['pK']:.2f}"
    return text
```

Programmatic templates have a real advantage over LLM-generated rationales: every fact in the training data is verifiable by construction.

---

## 8. Parallel schedule (20 hours wall-clock, 4 people)

### 8.1 Lanes

| Lane | Person | Hours 0–6 | Hours 6–12 | Hours 12–20 |
|---|---|---|---|---|
| **Data + labels** | P1 | Load `affinity_data.csv`; pick 2k MISATO subset; build pK lookup; upload to S3 | Vina baseline (using MISATO's structures); GBT-on-features baseline | Final eval, comparison table, ablations |
| **Features + facts** | P2 | Loader that returns `(channels, facts)` from the HDF5; works against `tiny_md.hdf5` | Run extraction on full subset; upload `features_v1.npz` to S3 | Optional 5th channel; assist P3 with debugging |
| **OpenTSLM fork** | P3 | Fork the repo; provision Python 3.12 env on SageMaker; scaffold `MISATOMDQADataset` with **dummy** tensors; sanity-train end-to-end (v1a only — no head yet) | Swap dummy → real features; train v1a (faithful, LoRA on Llama-3.2-1B SP); then add the ~40-line regression-head surgery and start v1b | 2–3 hyperparam runs on v1b (sweep `λ ∈ {0.1, 0.5, 1.0}`); select best of v1a and v1b; package both weights |
| **Rationales + verifier** | P4 | Lock 5-claim vocabulary; templater on synthetic fact dicts | Generate training rationales for real data (depends on P2); regex claim extractor + grounding function | Run verifier on P3 outputs; % verified report; demo examples |

### 8.2 Coordination beats

- **Integration checkpoints at H6, H12, H17.** P3 merges from each lane's branch.
- **One-hour rule.** If anyone is blocked >1 h, they ping the shared channel and someone context-switches.
- **AWS bottleneck.** Provision all four IAM users in hour 0 so no one is locked out.
- **Idle-GPU shutdown.** Set SageMaker Studio idle timeout to 30 min.

---

## 9. Hour 0 checklist — non-negotiable

These eight items lock at the kickoff and are not re-opened.

1. **Data contract.** `features: float32 (N, 100, 4)`, `pK: float32 (N,)`, `pdb_ids: str (N,)`, `facts: list[dict]`. Saved at `s3://<bucket>/features_v1.npz` + `facts_v1.jsonl`.
2. **Channel order.** `[ligand_rmsd, interaction_energy, distance, bSASA]`. Reorder later = silent bug.
3. **pK formula.** `pK = 9 - log10(nM)`, priority `Kd > Ki > IC50`.
4. **Claim vocabulary.** 5 types listed in §7.2, frozen.
5. **System count.** 2,000 for v1. (Can expand to 5k if budget permits; full 17k is overkill.)
6. **Base checkpoint.** `OpenTSLM/llama-3.2-1b-tsqa-sp` (smallest with SP weights already adapted to TS).
7. **Instance.** `ml.g5.xlarge` spot, idle-shutdown 30 min, all 4 users.
8. **Shared repo.** GitHub fork of `StanfordBDHG/OpenTSLM`. Branch-per-person. P3 owns `main`.

---

## 10. Baselines we have to beat

| Baseline | What it is | Difficulty to beat |
|---|---|---|
| AutoDock Vina | Docking score → affinity rank | Easy (headline) |
| Static-frame GNN | GNN trained on frame 0 of each trajectory, same labels, same split | Medium — this is the real test of "MD adds something" |
| Hand-engineered MD features + MLP | Mean RMSD, mean energy, mean dist, mean bSASA → MLP | Medium — this tests "the encoder adds something over averages" |
| `misato-affinity` GCN | Their published baseline, checkpoint shipped | Hard but optional |

If we beat Vina but not the static-frame GNN, MD added nothing. If we beat the static-frame GNN but not the MLP-on-averages, the encoder added nothing. Both internal checks must pass for the contribution to be real.

**Plus the internal v1a vs v1b check:** if v1b doesn't beat v1a by more than ~0.05 Pearson R, the regression-head extension wasn't worth its complexity and we recommend v1a in any production guidance.

---

## 11. Risks and honest limits

1. **10 ns ceiling.** Experimental ΔG reflects equilibrium; 10 ns of dynamics can't fully recover it. There's a hard floor on Pearson R from this.
2. **Tokenization noise on `Answer: 6.42` (v1a only).** Llama splits floats into ~3 tokens; numeric precision ceiling around ±0.1 pK on the string-parsed prediction. This is exactly why we also train v1b — the regression head bypasses tokenization. The v1a–v1b gap is informative: if it's small, tokenization wasn't the bottleneck.
3. **Selection bias in MISATO.** The 17 K systems were stable enough to simulate. Unstable / unbindable systems were filtered out. Our model only sees the easy regime.
4. **133 GB download.** P1 must start the download in hour 0 in the background; if the link is slow we may need to subset to 5 K systems streamed from Zenodo directly into the SageMaker job.
5. **Rationale faithfulness.** Free-form generation will make claims outside the closed vocabulary. The verifier marks those `unverifiable` (not contradicted). Reported % verified is over **verifiable** claims only.
6. **Topology files.** Only one system (`11gs`) has its AMBER topology locally. Per-residue claims in the rationale need topology; we deliberately scope claims to the four shipped channels for v1.

---

## 12. Decisions still open (need an answer before hour 0)

1. **AWS state.** Are SageMaker Studio + an S3 bucket already provisioned for all four users, or does setup happen inside the 20 h?
2. **Subset size.** 2 K (default), 5 K (richer), or full 17 K (much more compute)?
3. **Generated answer format.** Free-form `pK` ("Answer: 6.42") or quantized class ("Answer: bucket_22")? Default: free-form. (Note: v1b's regression head sidesteps this entirely; the answer string is still trained but only v1a parses it for the affinity prediction.)
4. **Rationale source.** Templated only (recommended) or LLM-rewritten for fluency (optional polish pass, no new facts introduced)?
5. **Eval split.** Use MISATO's `test_MD.txt` (1,612 systems) or `misato-affinity`'s `test_pairs.pickle` (163 pairs, ranking task)?

---

## 13. Deliverables

By hour 20:

- A reproducible SageMaker notebook that loads weights and produces (affinity, rationale, verification report) for any held-out MISATO PDB ID. Supports both v1a (string-parsed pK) and v1b (regression-head pK).
- **Two checkpoint sets** packaged on S3 (and optionally pushed to fresh HF repos):
  - v1a: LoRA + encoder + projector weights — the faithful variant.
  - v1b: LoRA + encoder + projector + **regression-head** weights — the hybrid variant.
- A one-page result writeup with the v1a vs v1b comparison front and centre: Pearson R, RMSE, % verified rationales, comparison to Vina + static-GNN + MLP-on-averages baselines, 3 worked examples (one easy / one hard / one failure mode).
- This document, updated with final numbers.

---

## Appendix A — file paths reference

All under `/home/mxlk/Documents/AIproject /` (note trailing space).

```
misato-dataset-master/
  data/MD/h5_files/tiny_md.hdf5              # 93 MB, 20 systems, full schema (smoke tests)
  data/MD/splits/{train,val,test}_MD.txt     # full PDB lists for the 17 K systems
  src/data/components/datasets.py            # reference for HDF5 access patterns
  src/data/processing/h5_to_traj.py          # how to reconstruct trajectories if needed

misato-affinity/
  data/affinity_data.csv                     # 19,443 rows: PDB → Kd/Ki/IC50
  configs/best_ckpts/                        # their GCN baselines (optional comparison)

OpenTSLM/
  src/opentslm/model/llm/OpenTSLMSP.py       # SP model
  src/opentslm/time_series_datasets/har_cot/HARCoTQADataset.py   # template to copy
  curriculum_learning.py                     # add stage6_misato here

QM (1).hdf5                                  # 328 MB, 19,413 ligands, optional ablation source
```

## Appendix B — pK conversion (Python)

```python
import math

def affinity_row_to_pK(row):
    """Convert a row from affinity_data.csv to a single pK label.

    Priority: Kd > Ki > IC50. Returns None if no usable value.
    pK = 9 - log10(value_in_nM).
    """
    for col in ("Kd (nM)", "Ki (nM)", "IC50 (nM)"):
        try:
            v = float(row[col])
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            return 9.0 - math.log10(v)
    return None
```

## Appendix C — key external references

- OpenTSLM paper: <https://arxiv.org/abs/2510.02410>
- OpenTSLM repo: <https://github.com/StanfordBDHG/OpenTSLM>
- OpenTSLM HF org: <https://huggingface.co/OpenTSLM> (base checkpoints)
- MISATO paper: *Nature Computational Science*, Siebenmorgen et al., 2024
- MISATO repo: <https://github.com/t7morgen/misato-dataset>
- MISATO Zenodo (133 GB MD HDF5): <https://zenodo.org/record/7711953>
- misato-affinity repo: <https://github.com/kierandidi/misato-affinity>
