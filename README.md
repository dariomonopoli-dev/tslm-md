# TSLM-MD

Agentic binding-affinity copilot for drug discovery — **first Time-Series Language Model applied to molecular dynamics**.

Built on [OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM) — specifically the [liu-jc Chronos-2 fork](https://github.com/liu-jc/OpenTSLM/tree/add-chronos2-encoder) directly recommended by the OpenTSLM team — gated cross-attention adapter over a frozen Llama-3.2-1B + Amazon's Chronos-2 as time-series encoder. Pretrained checkpoint: [`juncliu/llama-3.2-1b-ecg-flamingo-epoch-35`](https://huggingface.co/juncliu/llama-3.2-1b-ecg-flamingo-epoch-35). Data: [MISATO](https://github.com/sab148/MiSaTo-dataset) protein-ligand MD trajectories + per-frame energies.

ETH Agentic Systems Lab × AWS Hackathon — May 2026.

## Architecture in one diagram

```
PDB id → MISATO HDF5 → featurize() → [6 features × 30 frames] tensor
                                                │
                                                ▼
                            ┌───────────────────────────────────┐
                            │ OpenTSLMFlamingo                  │
                            │  CNNTokenizer → Perceiver →       │
                            │  gated cross-attention every N    │
                            │  layers of frozen Llama-3.2-1B    │
                            └───────────────────────────────────┘
                                                │
                                                ▼
                              "Answer: -8.4 kcal/mol. Confidence: high."
                                                │
                                                ▼
                                          agent(pdb_id)
                                                │
                              ┌─────────────────┼─────────────────┐
                              ▼                 ▼                 ▼
                       trained TSLM   deterministic        physics-based
                       prediction     grounded             verifier
                                      rationale            (independent)
                                                │
                                                ▼
                                  verdict: CONFIRMED | INCONCLUSIVE
```

The trained adapter (~50-200 M params) is the artifact. The agent loop wraps it with a verifier that abstains when prediction disagrees with independent physics — mirroring the SOC-agent prior-winner precedent.

## Quick start

Works on any single-GPU CUDA box. Tested target: **vast.ai A100 80GB or H100 80GB** (~$1-2/hr, recommended for VRAM headroom). Also works on a local A30 24GB.

```bash
git clone git@github.com:dariomonopoli-dev/tslm-md.git
cd tslm-md
bash scripts/setup_gpu.sh      # venv + clone OpenTSLM/MISATO + pip install + HF login + pre-warm cache
python scripts/dry_run.py      # the 30-min go/no-go test
```

If `dry_run.py` reports `✅ DRY-RUN PASSED`, kick off hour 0:

```bash
bash scripts/start_hour0.sh
```

If it fails, debug from the printed error (single most useful pre-build signal — better to find dep hell now than at hour 6).

## Pre-clock prep (run BEFORE the 24-hour clock starts)

The single biggest critical-path saver. Without this, the 133 GiB MISATO download eats hours 0-6 of the official timeline.

**Recommended (on the same vast.ai/cloud GPU box):**

```bash
bash scripts/download_misato_direct.sh
```

Datacenter networks pull from Zenodo at ~50-200 MB/s with `aria2c -x 8`. Total wall time ~10-60 min. Optionally `aws s3 sync` afterwards for durability.

**Fallback (only if your download endpoint is home Wi-Fi):**

```bash
bash scripts/download_misato_via_ec2.sh
```

Spins up a `c6i.xlarge` EC2 instance, downloads MISATO from Zenodo on AWS's pipe, pushes to S3. Total cost ~$0.30-$0.50.

## AWS surface

| Service | Role | When |
|---|---|---|
| **S3** (us-east-1 or us-west-2) | Stores MISATO + checkpoints | Pre-clock + always |
| **Bedrock** (Claude Haiku 4.5) | Second-opinion summariser in demo | Hour 20+ |
| **SageMaker Endpoint** | Serves trained checkpoint to Streamlit demo | Hour 18+ (optional) |
| **SageMaker Training Jobs** | Productionisation path | Pitch slide |

## Documents

- **Architecture spec** — `docs/superpowers/specs/2026-05-21-tslm-md-design.md`
- **Idea evaluation** — `docs/evaluation/idea-1-tslm-md.md` + comparison matrix
- **Phased plan + hour-gates** — inside the architecture spec, §5

## Repo layout

```
tslm_md/                package (installable via pip install -e .)
  featurize.py          raw HDF5 trajectory → [6, F] tensor   (REAL)
  prompts.py            pre/post prompt templates              (REAL)
  parse.py              regex → (affinity, confidence)         (REAL)
  dataset.py            MDCoTQADataset (OpenTSLM QADataset)    stub
  rationale.py          deterministic grounded summariser      stub
  agent.py              the multi-step loop                    stub
  verifier.py           physics-based independent verifier     stub
  eval.py               Pearson r + abstention metrics         stub
  train_stage6.py       plug into CurriculumTrainer            stub
  bedrock_summarizer.py optional Claude second-opinion         REAL (graceful fallback)

scripts/
  dry_run.py                       30-min go/no-go             FULL
  setup_gpu.sh                     one-shot env setup          FULL
  download_misato_direct.sh        direct Zenodo download      FULL (recommended)
  download_misato_via_ec2.sh       EC2→S3 fallback prefetch    FULL
  deploy_sagemaker_endpoint.sh     deploy trained model        stub
  preprocess_features.py           batch featurise → h5        stub
  build_training_targets.py        PDBbind → "Answer: X..."    stub
  train_gbm_baseline.py            R1 disproof experiment      stub
  train_cmapss_fallback.py         hour-14 insurance           stub
  start_hour0.sh                   kicks off real pipeline     stub

configs/
  stage6_md_cot.yaml               training config
  agent.yaml                       inference + verifier config

demo/
  app.py                           Streamlit live demo         stub

docs/
  superpowers/specs/   architecture spec
  evaluation/          team idea-comparison notes
```

## Hour-gates (from spec §5)

| Hour | Gate |
|---|---|
| 4 | Dataloader yields featurised batches on GPU |
| 8 | OpenTSLMFlamingo overfits a single batch (loss < 0.05) |
| 14 | Stage-6 training loss decreasing AND val Pearson > 0.15 |
| If hour 14 fails | Pivot to C-MAPSS architecture demo (insurance script runs in parallel from hour 6) |

## License

MIT (this repo). OpenTSLM is MIT. MISATO is Apache-2.0. PDBbind is research-only.
