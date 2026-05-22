"""Zero-shot evaluation of the pretrained OpenTSLM-SP checkpoint on MISATO.

Loads OpenTSLM/llama-3.2-1b-tsqa-sp (warm-start weights from the TSQA
curriculum) with NO additional fine-tuning, runs it against the MISATO
val/test splits, and reports affinity-prediction metrics in the same shape
as train_misato.py's per-epoch eval.

This is the "did fine-tuning help?" baseline. Compare its RMSE / Pearson R
to v1a's best epoch — the gap is how much MISATO fine-tuning added on top
of TSQA pretraining.

Also a sanity check on the trajectory encoder: if zero-shot Pearson R is
already > 0, the TSQA encoder is transferring some general TS->LLM skill to
MD. If it's near zero (likely — TSQA was ECG/Sleep/Accel/M4, not MD), then
all the predictive power comes from MISATO fine-tuning specifically.

Usage:
  # Quick check on val only (subset for speed)
  python eval_pretrained.py --splits val --subset 200

  # Full val + test
  python eval_pretrained.py --splits val test \\
      --output runs/pretrained_baseline.json

  # Same, plus save every generation for verify_rationale.py
  python eval_pretrained.py --splits test \\
      --output runs/pretrained_test.json \\
      --save-generations runs/pretrained_test_generations.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.time_series_datasets.misato.MISATOMDQADataset import MISATOMDQADataset

# Reuse helpers from the trainer.
from train_misato import (
    collate_misato, evaluate, metrics, parse_pK_from_text,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--warm-start", default="OpenTSLM/llama-3.2-1b-tsqa-sp",
                   help="HF repo. Maps internally to meta-llama/Llama-3.2-1B as base.")
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--subset", type=int, default=None,
                   help="Cap samples per split (for fast smoke testing).")
    p.add_argument("--output", type=Path, default=None,
                   help="Write metrics JSON to this path.")
    p.add_argument("--save-generations", type=Path, default=None,
                   help="Write {pdb_id, true_pK, pred_pK, rationale} JSONL per sample.")
    p.add_argument("--no-eval-shortcut", action="store_true",
                   help="Even without --save-generations, run a custom loop "
                        "(slower but useful if you want to inspect outputs).")
    return p.parse_args()


@torch.no_grad()
def eval_with_generations(model, dataset, split: str, batch_size: int,
                          max_new_tokens: int) -> tuple[dict, list[dict]]:
    """Like train_misato.evaluate but also returns the raw generations."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_misato, num_workers=0)
    model.eval()
    preds: list[float | None] = []
    trues: list[float] = []
    rationales: list[str] = []
    pdbs: list[str] = []

    for batch in tqdm(loader, desc=f"eval[{split}]"):
        gen = model.generate(batch, max_new_tokens=max_new_tokens, do_sample=False)
        for text, b in zip(gen, batch):
            preds.append(parse_pK_from_text(text))
            trues.append(float(b["pK"]))
            rationales.append(text)
            pdbs.append(b["pdb_id"])

    trues_arr = np.array(trues)
    preds_arr = np.array([np.nan if p is None else p for p in preds])
    result = {
        "split": split,
        "n": len(trues),
        "string_parse": metrics(preds_arr, trues_arr),
    }
    generations = [
        {"split": split, "pdb_id": p, "true_pK": float(t), "pred_pK": pr,
         "rationale": r}
        for p, t, pr, r in zip(pdbs, trues, preds, rationales)
    ]
    return result, generations


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available; CPU eval will be very slow.")

    print(f"loading pretrained: {args.warm_start}  device={device}")
    t0 = time.time()
    model = OpenTSLM.load_pretrained(
        args.warm_start, device=device, enable_lora=True,
    )
    print(f"model ready in {time.time() - t0:.1f}s")
    print("NOTE: this script does NOT fine-tune. Model is evaluated as-loaded.")

    all_results: dict = {
        "model": args.warm_start,
        "max_new_tokens": args.max_new_tokens,
        "subset": args.subset,
        "results": {},
    }
    all_generations: list[dict] = []

    capture = args.save_generations is not None or args.no_eval_shortcut

    SPLIT_MAP = {"val": "validation", "validation": "validation", "train": "train", "test": "test"}
    for split in args.splits:
        ds_split = SPLIT_MAP[split]
        print(f"\n=== split: {split} ===")
        ds = MISATOMDQADataset(split=ds_split, EOS_TOKEN=model.get_eos_token())
        if args.subset and args.subset < len(ds):
            ds = Subset(ds, list(range(args.subset)))
            print(f"  using subset of {len(ds)} samples")
        else:
            print(f"  full split: {len(ds)} samples")

        if capture:
            result, generations = eval_with_generations(
                model, ds, split, args.batch_size, args.max_new_tokens,
            )
            all_generations.extend(generations)
        else:
            result = evaluate(
                model, ds, args.batch_size, args.max_new_tokens, split,
            )

        all_results["results"][split] = result
        print(f"  {split}: {json.dumps(result)}")

    print("\n=== summary ===")
    print(json.dumps(all_results, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2))
        print(f"wrote metrics → {args.output}")

    if args.save_generations:
        args.save_generations.parent.mkdir(parents=True, exist_ok=True)
        with args.save_generations.open("w") as f:
            for g in all_generations:
                f.write(json.dumps(g) + "\n")
        print(f"wrote {len(all_generations)} generations → {args.save_generations}")
        print("Next: feed to verify_rationale.py to score grounding:")
        print(f"  python verify_rationale.py --predictions {args.save_generations} --split {args.splits[-1]}")


if __name__ == "__main__":
    main()
