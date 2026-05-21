"""Test loading the juncliu Chronos-encoder checkpoint into a fresh OpenTSLMFlamingo.

Reports how many keys match vs are missing/unexpected — tells us if the
architecture matches the published weights cleanly enough to fine-tune.

Usage:
    python scripts/test_checkpoint_load.py
"""

from __future__ import annotations

import argparse
import os
import glob
from pathlib import Path

import torch

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo


DEFAULT_CKPT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--juncliu--llama-3.2-1b-ecg-flamingo-epoch-35"
    "/snapshots/*/best_model.pt"
)


def find_ckpt(pattern: str) -> Path:
    matches = glob.glob(pattern)
    if not matches:
        raise SystemExit(f"no checkpoint matching {pattern}")
    return Path(matches[0])


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    ckpt = Path(args.checkpoint) if args.checkpoint else find_ckpt(DEFAULT_CKPT_GLOB)
    print(f"checkpoint: {ckpt}")

    print(f"instantiating model with encoder_type={args.encoder_type}...")
    m = OpenTSLMFlamingo(
        device=device,
        llm_id=args.llm_id,
        encoder_type=args.encoder_type,
    )

    print(f"loading state dict from {ckpt}...")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        print("  (state has model_state key, unwrapping)")
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        print("  (state has state_dict key, unwrapping)")
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise SystemExit(f"checkpoint does not contain a state-dict-like object (got {type(state)})")

    missing, unexpected = m.model.load_state_dict(state, strict=False)
    total = len(missing) + len(unexpected) + len(state)
    print(f"\nstate dict has {len(state)} keys")
    print(f"missing in model    : {len(missing)}")
    print(f"unexpected from ckpt: {len(unexpected)}")
    print(f"missing examples    : {missing[:5]}")
    print(f"unexpected examples : {unexpected[:5]}")

    if len(missing) < 50 and len(unexpected) < 50:
        print("\n[OK] looks like a clean fit — fine-tune should converge fast")
    elif len(missing) < 500 and len(unexpected) < 500:
        print("\n[WARN] some mismatch — fine-tune may still work but check examples")
    else:
        print("\n[FAIL] heavy mismatch — checkpoint and model architecture differ")
        print("       Likely cause: encoder_type or LLM mismatch")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="path to best_model.pt (default: auto-find juncliu cache)")
    p.add_argument("--llm-id", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--encoder-type", default="chronos2", choices=["chronos2", "cnn"])
    main(p.parse_args())
