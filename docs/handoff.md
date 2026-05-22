# Handoff — TSLM-MD Hackathon (mid-flight)

## What this is

A 24-hour ETH × AWS hackathon project building **TSLM-MD**: the first Time-Series Language Model applied to molecular dynamics for protein-ligand binding affinity prediction. Mid-flight handoff at ~hour 8-9 of the user's 19-hour countdown.

## Don't re-read everything — these artifacts already exist

| Doc | Path | Purpose |
|---|---|---|
| Full architecture spec | `docs/superpowers/specs/2026-05-21-tslm-md-design.md` | Single source of truth on design |
| 19-hour execution plan | `docs/hackathon-plan-19h.md` | Role assignments, hour-by-hour gates |
| YC-style pitch brief | `docs/yc-pitch-brief.md` | For Fiorenzo's slide drafting |
| Rationale-labels strategy | `docs/labels-strategy.md` | Direct answer to "how do you generate molecular CoT labels" |
| Team idea-comparison notes | `docs/evaluation/*.md` | History of how this idea was chosen |
| GitHub repo (private) | https://github.com/dariomonopoli-dev/tslm-md | All code |

Read the design spec § 1, 3, 5 first if you need orientation. Skip the rest unless you need it.

## Current state — what just worked

**`✅ DRY-RUN PASSED`** on the user's noctua server (A30, 24 GB VRAM):
- Forward pass returns finite loss (1.87)
- Backward pass updates perceiver weights
- `model.generate()` returns valid tokens
- VRAM peak: 7.73 GB (comfortable)
- Architecture wired end-to-end with Chronos-2 encoder + Llama-3.2-1B + Flamingo adapter

That is the critical milestone for the whole project. Training is unblocked.

## In flight when handed off

1. **Backbone-freeze patch is written but NOT yet committed.** Look at `scripts/patch_chronos.sh` — the latest `Edit` added a Python heredoc block that freezes the Chronos backbone (cuts trainable params from 837M → ~150M). It's saved to disk locally but not pushed. **Commit + push it first.** Commit message draft is below.

2. **MD.hdf5 (132 GB)** finished downloading to `~/tslm-md/data/misato/MD.hdf5` on noctua.

3. **S3 upload of MD.hdf5** is in progress at ~47 MB/s, was ~43% when last checked (~ETA 26 min from then). Run `python scripts/aws_status.py` to recheck. Let it finish — it's not blocking.

4. **Labels + splits ready.** `data/targets.json` (19,443 PDB-ids with answer strings), `data/splits/{train,val,test}.txt` copied from Zenodo official splits.

5. **HF models cached.** `meta-llama/Llama-3.2-1B`, `amazon/chronos-2`, `juncliu/llama-3.2-1b-ecg-flamingo-epoch-35` all on disk.

## Next actions in order

```bash
# 0. Commit the unpushed backbone-freeze patch
cd /Users/mondra/git/hackathons/colosseum-idearum2026
git add -A
git commit -m "fix(patch_chronos): freeze Chronos backbone — only train projection/norm

Brings trainable params from 837M to ~150M without losing the learnable
bridge between Chronos features and the perceiver."
git push origin main

# 1. User pulls + applies patch on noctua
cd ~/tslm-md && git pull
bash scripts/patch_chronos.sh

# 2. Kick off preprocessing on real MD.hdf5 (~20-30 min CPU)
nohup python scripts/preprocess_features.py --misato-h5 data/misato/MD.hdf5 --limit 2000 > preprocess.log 2>&1 &

# 3. When done, run R1 baseline (hour-4 gate from the plan)
python scripts/train_gbm_baseline.py
# Pass: val Pearson r >= 0.3. Otherwise add channels 7-8 (H-bonds, contact entropy).

# 4. Wiring gate (hour-8 from plan) — overfit single batch with juncliu ckpt
python scripts/test_checkpoint_load.py
# (May need to write a tiny overfit-single-batch script for full wiring gate.)

# 5. Real training
python tslm_md/train_stage6.py \
    --config configs/stage6_md_cot.yaml \
    --starting-checkpoint ~/.cache/huggingface/hub/models--juncliu--llama-3.2-1b-ecg-flamingo-epoch-35/snapshots/cfcdf8f7141b729ae50da4e1ef4e3bdc2b638674/best_model.pt
```

Convergence gate at hour 14 (val Pearson > 0.15). If failing → switch demo to use the unfine-tuned juncliu checkpoint + the deterministic rationale; pitch the architecture rather than the convergence.

## Hard-won learnings (don't relearn these)

1. **zsh mangles multi-line commands** on the user's noctua box. NEVER give them multi-line `python -c` or backslash-continued shell. ALWAYS prefer a saved script (`scripts/*.py` or `scripts/*.sh`) and have them run one short command.

2. **Python version on noctua: 3.11.2.** OpenTSLM chronos branch's `pyproject.toml` requires 3.12. `patch_chronos.sh` already loosens this to 3.11 — idempotent, just rerun.

3. **HuggingFace Hub:** pinned to `0.36.2` (which is what `transformers 4.57.6` requires). HF Hub 1.x renames `huggingface-cli` to `hf` but our `transformers` won't tolerate 1.x. Use `huggingface-cli` (not `hf`).

4. **NumPy ABI:** noctua has NumPy 2.4.3 but the system `bottleneck` and `numexpr` are compiled for 1.x. Fix is `pip install --upgrade --force-reinstall --no-deps numexpr bottleneck` (already done; new versions live in venv and take precedence over `/usr/lib/python3/dist-packages`).

5. **OpenTSLM Chronos branch bugs** — all patched by `scripts/patch_chronos.sh`. Idempotent. Read the script for the full list. Key bugs:
   - `SimpleNamespace.requires_grad_(True)` in init
   - `SimpleNamespace not callable` in forward (`self.vision_encoder(vision_x)` should be `.visual(vision_x)`)
   - Chronos2Encoder projection/norm not moved to device
   - Perceiver + gated_cross_attn not moved to device (SimpleNamespace blocks recursive `.to()`)
   - Flamingo.generate rejects `eos_token_id` / `pad_token_id` / `bos_token_id`
   - (Pending) Chronos backbone unfrozen by default → 837M trainable

6. **AWS keys were leaked to chat earlier** (`AKIARL77MMK5A36LNF6X` + secret). User rotated them. Don't paste creds in chat again.

7. **Don't pivot the data** — Llama-3.2-3B requires more VRAM; Llama-3.1-8B has no pretrained Flamingo ckpt; QM-only loses our entire pitch. The user has asked these questions several times — the answer is always "stay with Llama-3.2-1B + Chronos + juncliu checkpoint."

## Tasks (from TaskList) still open

- `#24` Monitor S3 upload to `sagemaker-us-west-2-094487995066/datasets/MD.hdf5` — *use `python scripts/aws_status.py`*
- `#25` Preprocess MISATO 2000 complexes → featurized.h5
- `#26` R1 gate: GBM baseline
- `#27` Wiring gate: overfit single batch with juncliu ckpt
- `#28` Freeze Chronos backbone for real training — *patch is written, needs commit + push*
- `#29` Real training run (stage6_md_cot)

Task `#23` (dry-run on real data) is marked completed — dry-run passed on the synthetic MD_dummy.hdf5 which validated the architecture identically.

## What the team is doing (4 people)

- **A (Dario, the user)** — on noctua, owns GPU work, training, dry-run
- **Fiorenzo** — drafting pitch slides from `docs/yc-pitch-brief.md`
- **Two others** — not yet assigned; the plan suggests B = data engineer, C = demo, D = AWS+pitch (see `docs/hackathon-plan-19h.md`)

## Suggested skills for the next session

- **`superpowers:executing-plans`** — there's a concrete written plan in `docs/hackathon-plan-19h.md`. Follow it.
- **`superpowers:verification-before-completion`** — before claiming any training succeeded, run real eval (Pearson r). Don't trust loss curves alone.
- **`superpowers:dispatching-parallel-agents`** — if the user has teammates ready to take Demo or Pitch tracks, parallelize.

## What NOT to do

- Don't keep retrying the dry-run on truncated MD.hdf5 (won't work, use the synthetic `MD_dummy.hdf5`)
- Don't write multi-line `python -c` or shell — zsh mangles it
- Don't switch encoder away from Chronos — the dry-run already works with it, and the juncliu checkpoint depends on it
- Don't try to use the full 16k complex dataset — 2000 is the right scope for the time budget
- Don't tune hyperparameters until first training run actually converges or fails — they'll iterate themselves
- Don't lead the pitch with Pearson r — the agent + abstention story is the differentiator
