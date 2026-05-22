# MoleMotion inference service

FastAPI backend that serves the trained TSLM (v1a + v1b), runs the regex
verifier, reconstructs PDB trajectories from MISATO HDF5, and orchestrates
the independent agent loop (Claude Opus 4.7 via OpenRouter).

This is a skeleton — implementations land per task. See the project task
list (`TaskList`) for what's wired vs. stubbed.

## Local dev

```bash
cp .env.example .env  # fill in OPENROUTER_API_KEY + OPENAI_API_KEY
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

`/health` returns immediately; `/predict` and `/evaluate*` 501 until
their owning task lands.

## Layout

```
app.py                 FastAPI routes (task #7, #9, #18, #6)
inference.py           TSLM loader + predict()                       (task #7)
verifier.py            Deterministic regex rationale verifier         (task #8)
hdf5_to_pdb.py         MISATO HDF5 → multi-MODEL PDB                  (task #9)
orchestrator.py        Agent loop                                     (task #17)
tools/
  splits.py            lookup_split, actual_pK_lookup                 (task #15)
  coords.py            cluster_poses, clash_check, hbond_persistence, (task #15)
                       per_residue_contacts
  chemistry.py         ligand_descriptors                             (task #15)
  physics.py           vina_rescore                                   (task #16)
rag/
  ingest.py            One-shot corpus build                          (task #11)
  store.py             rag_query with label filter                    (task #12)
llm/
  openrouter.py        Anthropic-compatible OpenRouter client         (task #14)
  pricing.py           Local spend computation                        (task #19)
prompts/
  system.md            §9.3 system prompt
  user_template.md     per-call template
```

## Why OpenRouter for the LLM and OpenAI for embeddings

OpenRouter routes to Claude Opus 4.7 (`anthropic/claude-opus-4-7`) using
the Anthropic Messages API shape — clean tool-use round-trips. OpenAI's
`text-embedding-3-small` is the cheapest competent embedding model for
the RAG corpus. The two are not interchangeable; do not wire OpenAI as
the judge.
