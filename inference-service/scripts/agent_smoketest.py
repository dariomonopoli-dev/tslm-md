"""End-to-end agent smoke test using the mock TSLM backend.

Tests the FULL real agent loop:
  - Claude via OpenRouter (real LLM)
  - RAG query against ChromaDB + OpenAI embeddings (real, if OPENAI_API_KEY set)
  - In-process tools (real — needs MD.hdf5 mounted for coord tools)
  - Regex verifier (real)
  - Persistent verdict cache (real)

What's mocked: only inference.predict() — returns a templated rationale +
pK from preprocessed/samples_test.jsonl. Lets you validate the agent
infrastructure without a trained model or GPU.

Usage:
    export OPENROUTER_API_KEY=sk-or-v1-...
    # optionally also:
    export OPENAI_API_KEY=sk-...     # for real RAG; without it, rag_query 500s
    export INFERENCE_BACKEND=mock
    export MOCK_NOISE=lossy           # 'lossy' breaks one numeric claim so
                                      # the verifier shows a contradiction
    python scripts/agent_smoketest.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

# Allow `python scripts/agent_smoketest.py` from inference-service/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference
from orchestrator import evaluate_agent


GREEN = "\033[32m"; RED = "\033[31m"; CYAN = "\033[36m"; DIM = "\033[2m"; RESET = "\033[0m"


# Default 3 PDBs from MISATO test split (16PK / 1A7C / 1ADO at top of samples_test.jsonl).
# Override via --pdbs A,B,C or --random 3.
DEFAULT_PDBS = ["16PK", "1A7C", "1ADO"]


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdbs", default=",".join(DEFAULT_PDBS),
                   help="Comma-separated PDB ids (default: 16PK,1A7C,1ADO)")
    p.add_argument("--random", type=int, default=0,
                   help="Pick N random PDBs from the test split instead")
    p.add_argument("--variant", default="v1b", choices=["v1a", "v1b"])
    p.add_argument("--max-steps", type=int, default=6)
    args = p.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print(f"{RED}OPENROUTER_API_KEY not set — cannot run agent loop{RESET}")
        sys.exit(1)

    os.environ.setdefault("INFERENCE_BACKEND", "mock")
    os.environ.setdefault("MOCK_NOISE", "lossy")

    # Boot the mock backend exactly like app.py would.
    inference.warm_up(
        os.getenv("CHECKPOINT_DIR", "/app/checkpoints"),
        os.getenv("TEST_SPLIT_PATH", "/app/data/test_MD.txt"),
    )
    loaded = inference.variants_loaded()
    if not loaded:
        print(f"{RED}mock backend boot failed — no samples loaded{RESET}")
        sys.exit(1)
    all_ids = inference.list_pdb_ids()
    print(f"{CYAN}backend={inference.backend()}  variants={loaded}  "
          f"test_split={len(all_ids)} PDBs{RESET}\n")

    if args.random > 0:
        random.seed(42)
        pdbs = random.sample(all_ids, min(args.random, len(all_ids)))
    else:
        pdbs = [p.strip() for p in args.pdbs.split(",") if p.strip()]

    failures = 0
    for pdb in pdbs:
        if not inference.is_in_test_split(pdb):
            print(f"{DIM}[{pdb}] not in test split, skipping{RESET}")
            continue

        print(f"{CYAN}--- {pdb} ---{RESET}")
        pred = inference.predict(pdb, args.variant)
        print(f"  mock prediction: pK={pred['pK']}  hidden_pK={pred['hidden_pK']}")
        print(f"  rationale: {pred['rationale'][:120]}{'…' if len(pred['rationale']) > 120 else ''}")

        try:
            verdict, trace = await evaluate_agent(
                pdb_id=pdb,
                model_pK=pred["pK"],
                rationale=pred["rationale"],
                variant=args.variant,
                max_steps=args.max_steps,
            )
        except Exception as e:
            print(f"  {RED}agent crashed: {type(e).__name__}: {e}{RESET}")
            failures += 1
            continue

        rec = verdict.get("recommendation", "?")
        tool_calls = verdict.get("agent_trace", {}).get("tool_calls", 0)
        latency = verdict.get("agent_trace", {}).get("latency_ms", 0)
        in_tok = verdict.get("agent_trace", {}).get("input_tokens", 0)
        out_tok = verdict.get("agent_trace", {}).get("output_tokens", 0)

        ok = (
            rec in ("trust", "review", "discard")
            and tool_calls >= 1
            and isinstance(verdict.get("scores"), dict)
        )
        sym = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {sym} verdict={rec}  tool_calls={tool_calls}  "
              f"latency={latency/1000:.1f}s  tokens={in_tok}/{out_tok}")

        # Dump the trace tool names for quick eyeball
        used = [s["tool"] for s in trace]
        print(f"  tools used: {used}")

        if not ok:
            failures += 1
            print(f"  {DIM}full verdict:{RESET}")
            print("  " + json.dumps(verdict, indent=2).replace("\n", "\n  "))

        print()

    if failures:
        print(f"{RED}{failures} failure(s){RESET}")
        sys.exit(1)
    print(f"{GREEN}agent smoke test passed{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
