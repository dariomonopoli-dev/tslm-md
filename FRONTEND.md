# Frontend — MD-Trajectory Binding Affinity Demo

**A Gradio app on a Hugging Face Space that lets anyone pick a PDB ID, watch the trajectory in 3D, and see the model's predicted affinity + verified rationale.**

One-line pitch: the model deliverable is a checkpoint, but the *credibility* deliverable is a public link a recruiter or reviewer can click and see the movie-plus-rationale framing actually work end-to-end. This document specifies that frontend.

Companion to [PROJECT_BRIEF.md](./PROJECT_BRIEF.md). Read that first.

---

## 1. Goal

A public-facing, live-inference demo with three properties:

1. **Shows the "movie" framing visually** — frame slider drives a synchronised 3D viewer and time-series cursor, so the dynamics are not abstract.
2. **Surfaces the v1a vs v1b ablation directly** — a toggle swaps which checkpoint generated the prediction and rationale, making the contribution legible to non-specialists.
3. **Makes the grounding claim verifiable in the UI** — each rationale claim is color-coded (verified / contradicted / unverifiable) against the underlying channel data, so visitors can audit the model's reasoning by eye.

The frontend is **demo-grade**, not product-grade. No auth, no usage tracking, no multi-user state. One container, sleeps when idle.

---

## 2. Audience and deployment target

| Decision | Value |
|---|---|
| Primary audience | External (HF Space link in a paper supplement, on LinkedIn, in recruiter messages) |
| Deployment | Hugging Face Space, GPU `t4-small` (~$0.40/hr, sleeps after 30 min idle) |
| Inference mode | Live — checkpoint loaded at startup, forward pass on demand |
| Auth | None — public Space |
| URL shape | `huggingface.co/spaces/<org>/md-trajectory-affinity` |
| Polish bar | Looks credible next to other research demo Spaces. Not pixel-perfect. |

The HF Space pairs naturally with the v1a / v1b weight repos already in the brief's §13 deliverables — the Space pulls them with `huggingface_hub.snapshot_download` at boot.

---

## 3. Architecture — how the UI plugs into the model

```
                    ┌─────────────────────────────────────────┐
                    │   Hugging Face Space (t4-small GPU)     │
                    │                                         │
   user browser ──▶ │   Gradio 5.x app                        │
                    │   ├── PDB-ID dropdown (1,612 test IDs)  │
                    │   ├── 4-channel plotly (frame cursor)   │
                    │   ├── 3Dmol.js (multi-MODEL PDB)        │
                    │   ├── prediction panel (v1a/v1b toggle) │
                    │   └── rationale + verifier badges       │
                    │           │                             │
                    │           ▼                             │
                    │   inference.py                          │
                    │   ├── load_checkpoints() at boot        │
                    │   │     v1a: LoRA + encoder + projector │
                    │   │     v1b: same + regression head     │
                    │   ├── predict(pdb_id, variant) ─┐       │
                    │   └── hdf5_to_pdb(pdb_id) ──┐   │       │
                    │                             │   │       │
                    │   features_v1.npz (S3 mirror, ~200 MB)  │
                    │   facts_v1.jsonl   (same, ~5 MB)        │
                    └─────────────────────────────────────────┘
```

The UI does not load the 133 GB MISATO HDF5. It loads the **precomputed feature pack** P2 already produces for training (brief §8.1, lane 2), which contains the 4 channels per system. For the 3D viewer it additionally needs per-system atom coordinates — see §6.

---

## 4. Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  MD-Trajectory Binding Affinity                          [v1a │ v1b]│
│  "We read the movie, not the snapshot."                              │
├───────────────────────────────────┬──────────────────────────────────┤
│  PDB ID  [1A1B ▼]      [Predict]  │  3D pocket view                  │
│                                   │  ┌────────────────────────────┐  │
│  ─── Prediction ───               │  │                            │  │
│  Predicted pK   6.42              │  │   ligand in pocket         │  │
│  Actual pK      6.31              │  │   (frame 47 / 100)         │  │
│  |Δ|            0.11              │  │                            │  │
│  Variant        v1b (hybrid)      │  │   [protein cartoon + lig.] │  │
│                                   │  │                            │  │
│  ─── Rationale ───                │  └────────────────────────────┘  │
│  During the trajectory the        │  ◀  ●━━━━━━━━━━━━━━  ▶  47/99    │
│  interaction energy averages      │  [▶ play]  [⟳ loop]              │
│  -37.2 kcal/mol                   ├──────────────────────────────────┤
│  ✓ verified (mean=-37.21)         │  Per-frame channels              │
│  and stabilises after frame 20.   │  ┌────────────────────────────┐  │
│  ✓ verified (slope flat after 20) │  │ RMSD       ╱╲    ╱╲        │  │
│  Ligand RMSD remains under        │  │ energy   ╱  ╲╱╲╱  ╲        │  │
│  2.5 Å                            │  │ distance ─────────         │  │
│  ✗ contradicted (max=3.10)        │  │ bSASA    ▔▔▔▔▔▔▔▔          │  │
│  with buried SASA holding above   │  │              ┊             │  │
│  500 Å².                          │  │       cursor=47            │  │
│  ✓ verified (min=518)             │  └────────────────────────────┘  │
│  The pose is stable.              │                                  │
│  ? unverifiable (no claim type)   │                                  │
│                                   │                                  │
│  Verified  3 / 4   (75%)          │                                  │
│  Show baselines ▼                 │                                  │
└───────────────────────────────────┴──────────────────────────────────┘
```

Interactions:

- **Frame slider** is the spine. It drives the 3D viewer's MODEL index *and* the vertical cursor on the time-series plot. This is what makes the "movie" framing tangible.
- **v1a / v1b toggle** in the header re-runs prediction (or pulls cached result) for the other variant and updates the prediction panel + rationale + verifier badges. Side-by-side comparison via two Spaces tabs if requested.
- **Claim badges** sit *inline* in the rationale text (not in a separate report panel). Hover shows the exact channel value being checked.
- **"Show baselines"** expands a small table comparing predicted pK against Vina, static-frame GNN, and MLP-on-averages for the selected system. Pulled from a precomputed JSON.

---

## 5. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| App framework | **Gradio 5.x** | One file, Python-native, HF Space integration, supports custom HTML for 3D embed. Streamlit was the alternative but Gradio's `gr.HTML` + JS interop is cleaner for the 3Dmol bridge. |
| Plots | **Plotly** via `gr.Plot` | Native zoom/pan, shared x-axis cursor, exports to PNG for the writeup. Matplotlib was tempting but lacks interactive cursor. |
| 3D viewer | **3Dmol.js** via CDN, embedded in `gr.HTML` | Renders multi-MODEL PDB strings with built-in animation (`viewer.animate({loop: 'forward'})`). Auto-bonds by distance — no topology file required. NGL Viewer was the alternative but its Python wrapper is Jupyter-tied. |
| Model loading | `huggingface_hub.snapshot_download` + the same `OpenTSLMSP` load path used in training | Reuses P3's code. No new model glue. |
| Feature serving | Local `.npz` + `.jsonl` baked into the Space image (downloaded from S3 at build time) | 200 MB fits in a Space; avoids cold-start S3 calls per request. |
| Deployment | HF Space, `app.py` + `requirements.txt` + `README.md` (HF metadata block) | Git push → Space rebuilds. No Docker hand-rolling. |

Versions to pin (matches the training env in `OpenTSLM/pyproject.toml`):

```
python>=3.12
torch>=2.9
transformers>=4.57
peft>=0.18
gradio>=5.0
plotly>=5.20
huggingface_hub>=0.25
h5py>=3.11      # only needed if we keep HDF5 fallback
numpy>=2.0
```

---

## 6. The risky piece — HDF5 → PDB reconstruction

The 3D viewer needs a multi-MODEL PDB string per system. We do **not** have AMBER topology files locally (brief §11.6 — only `11gs` ships one). The plan is to synthesize a PDB from the HDF5 itself, using fields verified in the brief's §4.1 schema:

```
trajectory_coordinates      (100, N_atoms, 3)   float64
atoms_element               (N_atoms,)          str
atoms_residue               (N_atoms,)          int (residue sequence number)
atoms_number                (N_atoms,)          int (atomic number)
molecules_begin_atom_index  (3,)                int (protein/ligand/water boundaries)
```

We have enough to emit a valid PDB record per atom:

```
ATOM  {serial:5d}  {name:<4s}{resname:>3s} {chain:1s}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}
```

with conventions:

- `chain = "A"` for atoms before `molecules_begin_atom_index[1]` (protein), `"L"` for ligand atoms (between `[1]` and `[2]`), `"W"` for water (>= `[2]` — usually filtered out for visual clarity).
- `resname = "UNK"` if we don't have residue names; 3Dmol still renders fine. (Stretch: parse residue names from `atoms_type` AMBER nomenclature.)
- `name = atoms_element + index-within-residue`, e.g. `"C1"`, `"N2"`. 3Dmol doesn't care about the name string for rendering.
- Wrap each frame in `MODEL n` / `ENDMDL`.

**Risk: residue name vacancy.** Without residue names, 3Dmol's `cartoon` representation may not render correctly — it relies on canonical aa codes. Fallback: render protein as `line` or `tube`, ligand as `stick`. Less pretty but unambiguous. Decide at hour 0 of the UI lane; have both ready.

**Risk: atom count.** Some MISATO systems are >10,000 atoms. 100 frames × 10,000 atoms × ~80 chars per PDB line ≈ 80 MB string. Browser will choke. Mitigations, in order:

1. Drop waters before emitting (saves 60–80%).
2. For the 3D viewer, render only every 5th frame (20 frames instead of 100). The plot still uses all 100.
3. Reduce coordinate precision to 2 decimals.

Combined, this gets us under 5 MB per system. Acceptable.

Reference code lives at `OpenTSLM/spaces/md-trajectory-affinity/hdf5_to_pdb.py` (~80 lines, written fresh).

---

## 7. Component breakdown

```
spaces/md-trajectory-affinity/
  app.py                  # Gradio entrypoint (~200 lines)
  inference.py            # load v1a/v1b checkpoints, predict(), generate_with_pK() (~120 lines)
  hdf5_to_pdb.py          # multi-MODEL PDB synthesis from HDF5 (~80 lines)
  verifier.py             # imports P4's claim extractor + grounding fn (~30 lines wrapper)
  assets/
    threedmol_embed.html  # template with {pdb_string} placeholder + JS animation glue
    style.css             # claim-badge colors, layout tweaks
  data/
    features_v1.npz       # baked at build time from S3
    facts_v1.jsonl
    baselines.json        # precomputed Vina + static-GNN + MLP-on-averages predictions
  requirements.txt
  README.md               # HF Space metadata block (title, emoji, SDK=gradio, sdk_version)
```

### 7.1 `app.py` skeleton

```python
import gradio as gr
from inference import predict, list_pdb_ids
from hdf5_to_pdb import build_pdb_string
from verifier import verify_rationale

def run(pdb_id: str, variant: str):
    result = predict(pdb_id, variant=variant)         # {pK, rationale, hidden_pK}
    claims = verify_rationale(result["rationale"], pdb_id)
    pdb_str = build_pdb_string(pdb_id)
    channels_fig = plot_channels(pdb_id, cursor=0)
    viewer_html = render_3dmol(pdb_str)
    return result["pK"], format_rationale(result["rationale"], claims), \
           channels_fig, viewer_html, claims_summary(claims)

with gr.Blocks(css="assets/style.css", title="MD-Trajectory Affinity") as demo:
    gr.Markdown("# MD-Trajectory Binding Affinity")
    with gr.Row():
        pdb = gr.Dropdown(list_pdb_ids(), label="PDB ID")
        variant = gr.Radio(["v1a", "v1b"], value="v1b", label="Variant")
        run_btn = gr.Button("Predict", variant="primary")
    with gr.Row():
        with gr.Column():
            pk_out = gr.Number(label="Predicted pK")
            rationale_out = gr.HTML(label="Rationale (verified)")
            summary_out = gr.Markdown()
        with gr.Column():
            viewer_out = gr.HTML(label="3D pocket view")
            frame_slider = gr.Slider(0, 99, value=0, step=1, label="Frame")
            channels_out = gr.Plot(label="Per-frame channels")
    run_btn.click(run, [pdb, variant], [pk_out, rationale_out, channels_out, viewer_out, summary_out])
    frame_slider.change(update_cursor, [pdb, frame_slider], [channels_out, viewer_out])

demo.launch()
```

### 7.2 Frame-cursor JS bridge

3Dmol does not expose its viewer to Python callbacks directly. We use a hidden `gr.Number` that JS reads on change:

```html
<script>
  let viewer = $3Dmol.createViewer('viewer-div');
  viewer.addModelsAsFrames(`{{pdb_string}}`, 'pdb');
  viewer.setStyle({}, {cartoon: {color: 'spectrum'}, stick: {hidden: false}});
  viewer.zoomTo();
  window.setFrame = function(idx) { viewer.setFrame(idx); viewer.render(); };
</script>
```

Python side: `frame_slider.change(js="(f) => { window.setFrame(f); return f; }", ...)`. The cursor on the plot updates server-side via the same callback. This is the only piece of custom JS — keep it small.

---

## 8. Schedule integration (the cost)

The brief's 4-lane schedule has no UI lane. Adding ~8 hours of frontend means cutting or parallelising somewhere. Three viable plans:

| Plan | What changes | Verdict |
|---|---|---|
| **A. P4 parallelises (recommended)** | P4 starts Gradio shell with mocked predictions at H10–H14 alongside rationale work; swaps in real model at H16–H20. | Cleanest. Mock-first derisks the JS bridge. All baselines preserved. |
| **B. Drop the misato-affinity GCN baseline** | P1 redirects H12–H20 from "comparison to their GCN" to UI build. | Loses one comparison in the writeup (the brief already flags it "optional"). |
| **C. Punt UI to a follow-up day** | Hit the 20 h model deliverable; build UI in a 21–28 h overflow. | Honest about scope; loses the public-link-on-day-one moment. |

**Recommended: Plan A.** Revised lane-4 schedule:

| Hours | P4 work |
|---|---|
| 0–6 | Lock 5-claim vocabulary; templater on synthetic facts. *(unchanged)* |
| 6–10 | Generate training rationales for real data; regex claim extractor + grounding function. *(unchanged but tightened from H6–12)* |
| 10–14 | **Gradio shell with mocked predict() returning a fixed example.** Layout, plotly cursor, 3Dmol embed, JS bridge — all live but on stub data. |
| 14–16 | Run verifier on P3 outputs; % verified report. *(unchanged from §8.1)* |
| 16–18 | Swap mock predict() for real inference. Bake `features_v1.npz` and `facts_v1.jsonl` into the Space. |
| 18–20 | Deploy to HF Space; smoke-test on 5 systems; record 3 demo examples for the writeup. |

P3 hands off checkpoints to P4 at H16 (same as the original integration checkpoint). No new coordination beats required.

---

## 9. Hour-0 checklist additions

These extend brief §9, not replace it.

9. **HF org and Space name.** Create the Space at `huggingface.co/spaces/<org>/md-trajectory-affinity` in hour 0 so the URL is reservable and CI can push to it. Gate behind an HF token in repo secrets.
10. **GPU vs CPU Space.** Start on CPU `basic` (free) for the mock-first phase. Upgrade to `t4-small` only at H16 when real inference goes in. Saves ~$5 across the sprint.
11. **PDB ID list.** The dropdown lists the **test split only** (1,612 IDs from `test_MD.txt`), to avoid demoing on training data by accident. Hardcoded at build time from the split file.
12. **Disclaimer copy.** A one-liner at the bottom: *"Demo-grade artifact. 10 ns of MD cannot fully resolve experimental binding affinity — see [PROJECT_BRIEF.md §11](./PROJECT_BRIEF.md)."* Honesty is a feature.

---

## 10. Risks and limits

1. **3Dmol fails on a malformed PDB string.** Mitigation: unit-test `hdf5_to_pdb` on the 20 tiny-MD systems before the real subset arrives. Visual inspection in a notebook with `py3Dmol`. If it breaks, fall back to a static frame-0/frame-50/frame-99 triptych (still informative).
2. **Cold start time.** Loading Llama-3.2-1B + LoRA + encoder takes ~30 s on a t4-small. First visitor after sleep waits. Mitigation: a "loading model…" splash; nothing to do about the GPU wake itself.
3. **GPU cost runaway.** A linked-in post that goes mildly viral could run the Space 24/7. Mitigation: HF Space sleep timer is non-negotiable at 30 min idle. Worst case (24 h on): ~$10. Acceptable.
4. **Out-of-vocabulary PDB IDs.** Users can only pick from the test split; no "upload your own HDF5" path in v1 (would require AMBER reproc — out of scope).
5. **Claim-coloring depends on the verifier shipping on time.** If P4 is late, ship v0 of the UI with the rationale uncolored (just the text + the %-verified summary). Better than blocking deploy.
6. **HF Space build flakes.** Spaces occasionally fail to build on large `requirements.txt`. Mitigation: pin every dep; smoke-build at H10 even with the mock to flush out env issues.

---

## 11. What this UI does *not* do

Scoped out, deliberately:

- No user accounts, no saved prediction history.
- No upload-your-own-MD endpoint. (Would need a queued AMBER prep pipeline.)
- No per-residue claims in the rationale, hence no per-residue highlighting in the 3D viewer. The 5-claim vocabulary in brief §7.2 is whole-trajectory only.
- No mobile-optimised layout. The 3D viewer is desktop-first; phones get a "best viewed on desktop" notice.
- No comparison to off-the-shelf docking tools beyond the precomputed Vina baseline.
- No fine-tuning UI, no "retrain on my data" button. The Space is read-only.

If any of these become demands later (e.g., a reviewer asks for per-residue claims), they go in a post-sprint v2 doc, not this one.

---

## 12. Decisions still open

1. **HF org name.** Personal account vs new team org? Recruiter-facing means it should look institutional — recommend a new org with all four contributors as members.
2. **Cache strategy.** Per-PDB-ID predictions are deterministic. Cache them in an in-memory dict (resets on Space sleep) or persist to a JSON sidecar? Default: in-memory; persist only if cold-start latency becomes the bottleneck.
3. **Side-by-side v1a/v1b view.** Today: a toggle that re-runs. Stretch: two columns showing both predictions at once. Costs ~1 hour to add, but makes the ablation pop. Recommend: build the toggle first, add side-by-side only if H18–20 has slack.
4. **Telemetry.** Anonymous click counts on which PDB IDs visitors pick would inform a v2. But adding any analytics conflicts with the "no auth, no tracking" stance. Default: none.

---

## 13. Deliverables (the UI's own §13)

By hour 20:

- A live HF Space at `huggingface.co/spaces/<org>/md-trajectory-affinity`, sleeping when idle, waking on visit.
- Source under `spaces/md-trajectory-affinity/` in the shared repo, reproducible with `gradio app.py` locally given the HF tokens.
- A README on the Space page with: one-paragraph intro, link back to PROJECT_BRIEF.md, three worked examples (easy / hard / failure mode — same three as the writeup), citation block.
- Three short screen-recordings (15–30 s each) for the writeup and LinkedIn post: (a) frame slider driving the 3D viewer, (b) v1a vs v1b toggle on the same system, (c) a contradicted claim being caught by the verifier badge.

---

## Appendix A — minimal `hdf5_to_pdb` reference

```python
import h5py
import numpy as np

ELEMENT_MASSES = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "P": 15}

def build_pdb_string(pdb_id: str, h5_path: str,
                     drop_water: bool = True,
                     stride: int = 5,
                     coord_decimals: int = 2) -> str:
    """Synthesize a multi-MODEL PDB string from a MISATO HDF5 system.

    Args:
        pdb_id: top-level group name in the HDF5.
        stride: keep every `stride`-th frame (default 5 → 20 frames of 100).
        drop_water: omit atoms past molecules_begin_atom_index[2].
    """
    with h5py.File(h5_path, "r") as f:
        g = f[pdb_id]
        coords = g["trajectory_coordinates"][:]      # (100, N, 3)
        elements = g["atoms_element"][:].astype(str) # (N,)
        residues = g["atoms_residue"][:]             # (N,)
        bounds = g["molecules_begin_atom_index"][:]  # (3,)

    n_atoms = coords.shape[1]
    keep_atoms = np.ones(n_atoms, dtype=bool)
    if drop_water and len(bounds) >= 3:
        keep_atoms[bounds[2]:] = False
    atom_idx = np.where(keep_atoms)[0]

    chains = np.where(np.arange(n_atoms) < bounds[1], "A", "L")
    frames = range(0, coords.shape[0], stride)

    lines = []
    for model_i, frame in enumerate(frames, start=1):
        lines.append(f"MODEL     {model_i:4d}")
        for i, ai in enumerate(atom_idx, start=1):
            x, y, z = coords[frame, ai]
            el = elements[ai].strip()
            lines.append(
                f"ATOM  {i:5d} {el+str(i%100):<4s} UNK {chains[ai]:1s}"
                f"{int(residues[ai]):4d}    "
                f"{x:8.{coord_decimals}f}{y:8.{coord_decimals}f}{z:8.{coord_decimals}f}"
                f"  1.00  0.00          {el:>2s}"
            )
        lines.append("ENDMDL")
    return "\n".join(lines)
```

~50 lines. Test against `data/MD/h5_files/tiny_md.hdf5` before relying on it.

---

## Appendix B — HF Space `README.md` metadata

```yaml
---
title: MD-Trajectory Binding Affinity
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
short_description: Watch the movie, predict the pK, verify the rationale.
---
```

Edit `emoji` only if the user explicitly wants the visual cue removed — HF Spaces convention is to include one.

---

## Appendix C — links

- PROJECT_BRIEF.md (this doc's parent) — `./PROJECT_BRIEF.md`
- Gradio docs — <https://www.gradio.app/docs>
- 3Dmol.js — <https://3dmol.csb.pitt.edu/>
- HF Spaces GPU pricing — <https://huggingface.co/pricing#spaces>
- OpenTSLM Space examples (for layout reference) — <https://huggingface.co/OpenTSLM>
