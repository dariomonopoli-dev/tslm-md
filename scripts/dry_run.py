"""TSLM-MD dry-run: end-to-end go/no-go before the 24-hour clock starts.

Confirms in ~30 minutes (assuming HF cache is pre-warmed):
  1. Featurisation works on the bundled MISATO example
  2. OpenTSLMFlamingo loads and fits in A30 VRAM
  3. Forward pass returns finite loss
  4. Backward pass updates the perceiver weights
  5. generate() returns valid tokens

If any step FAILs, diagnose now. Better to find dep hell here than at hour 6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import torch

from tslm_md.featurize import featurize, F_SUB, N_CHANNELS
from tslm_md.prompts import build_prompts, channel_descriptors


def step(n: int, name: str) -> None:
    print(f"\n=== STEP {n}: {name} ===")


def fail(msg: str) -> None:
    print(f"\n❌ FAIL: {msg}")
    sys.exit(1)


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device != "cuda":
        print("⚠ Not on CUDA. The dry-run will still run, but VRAM checks are skipped.")

    # -------- STEP 1
    step(1, "Load bundled MISATO sample")
    h5_path = Path(args.misato_h5)
    if not h5_path.exists():
        fail(
            f"{h5_path} not found.\n"
            f"   Make sure setup_a30.sh ran and cloned third_party/MiSaTo-dataset."
        )
    with h5py.File(h5_path, "r") as f:
        # PDB ids in MISATO are stored lowercase; accept either case here.
        if args.pdb_id in f:
            resolved_id = args.pdb_id
        elif args.pdb_id.lower() in f:
            resolved_id = args.pdb_id.lower()
        elif args.pdb_id.upper() in f:
            resolved_id = args.pdb_id.upper()
        else:
            available = list(f.keys())[:5]
            fail(f"PDB id '{args.pdb_id}' not in {h5_path}. Available: {available}")
        g = f[resolved_id]
        coords_shape = g["trajectory_coordinates"].shape
        mol_begin = g["molecules_begin_atom_index"][:]
        print(f"  resolved PDB id              = {resolved_id}")
        print(f"  trajectory_coordinates.shape = {coords_shape}")
        print(f"  molecules_begin_atom_index   = {mol_begin}")
        print(f"  ligand starts at atom index  = {int(mol_begin[-1])}")
        if coords_shape[-1] != 3:
            fail(f"expected last dim 3, got {coords_shape[-1]}")

    # -------- STEP 2
    step(2, f"Featurise trajectory → [{N_CHANNELS}, {F_SUB}]")
    with h5py.File(h5_path, "r") as f:
        feats = featurize(f[resolved_id])
    print(f"  features.shape = {tuple(feats.shape)}")
    print(f"  per-channel mean = {[round(x, 3) for x in feats.mean(dim=1).tolist()]}")
    print(f"  per-channel std  = {[round(x, 3) for x in feats.std(dim=1).tolist()]}")
    if feats.shape != (N_CHANNELS, F_SUB):
        fail(f"expected shape ({N_CHANNELS}, {F_SUB}), got {tuple(feats.shape)}")
    if not torch.isfinite(feats).all():
        fail("featurisation produced NaN or Inf")

    # -------- STEP 3
    step(3, "Build single-item OpenTSLM batch dict")
    pre, post = build_prompts(args.pdb_id)
    dummy_answer = "Answer: -7.2 kcal/mol. Confidence: medium."
    # Chronos encoder is univariate per chunk — pass 6 channels as 6 chunks
    # with one descriptor each (mirrors OpenTSLM ECG-QA 12-lead pattern).
    batch_item = {
        "time_series": feats,                                   # [6, 30] = 6 chunks x 30 frames
        "time_series_text": channel_descriptors(),              # 6 descriptors, one per chunk
        "pre_prompt": pre,
        "post_prompt": post,
        "answer": dummy_answer,
    }
    print(f"  keys: {list(batch_item.keys())}")
    print(f"  time_series dtype: {batch_item['time_series'].dtype}")

    # -------- STEP 4
    step(4, "Instantiate OpenTSLMFlamingo + check VRAM")
    try:
        from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
    except ImportError as e:
        fail(
            f"cannot import OpenTSLMFlamingo: {e}\n"
            f"   Did you run `pip install -e third_party/OpenTSLM`?"
        )

    model = OpenTSLMFlamingo(
        llm_id=args.llm_id,
        device=device,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        encoder_type=args.encoder_type,
    )
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"  trainable params: {trainable:,} ({trainable / 1e6:.1f} M)")
    if device == "cuda":
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM allocated: {alloc_gb:.2f} GB")
        if alloc_gb >= 22.0:
            fail(f"VRAM {alloc_gb:.2f} GB exceeds A30 budget (24 GB - 2 GB headroom)")
    if not (50e6 <= trainable <= 500e6):
        print(f"  ⚠ trainable params outside expected band [50M, 500M] — inspect.")

    # -------- STEP 5
    step(5, "Forward pass")
    input_ids, images, attention_mask, labels = model.pad_and_apply_batch(
        [batch_item], include_labels=False
    )
    print(f"  input_ids.shape = {tuple(input_ids.shape)}")
    print(f"  images.shape    = {tuple(images.shape)}")
    if labels is None:
        fail("labels are None — pad_and_apply_batch did not populate training targets")

    out = model.model(
        vision_x=images,
        lang_x=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = getattr(out, "loss", None)
    if loss is None and isinstance(out, tuple):
        loss = out[0]
    if loss is None:
        fail(f"could not extract loss from forward output (type={type(out).__name__})")
    print(f"  loss = {loss.item():.4f}")
    if not torch.isfinite(loss):
        fail("loss is not finite")
    if not loss.requires_grad:
        fail("loss does not require grad")

    # -------- STEP 6
    step(6, "Backward pass + AdamW step → check perceiver weights changed")
    trainable_params = [p for p in model.model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=1e-4)

    perc_snapshot = None
    for n, p in model.model.named_parameters():
        if "perceiver" in n and p.requires_grad:
            perc_snapshot = (n, p.detach().clone())
            break
    if perc_snapshot is None:
        print("  ⚠ no perceiver parameter found — skipping weight-change check")

    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)

    if perc_snapshot is not None:
        n, before = perc_snapshot
        after = dict(model.model.named_parameters())[n]
        diff = (after.detach() - before).abs().max().item()
        print(f"  perceiver weight max-change: {diff:.6f}")
        if diff <= 0.0:
            fail("perceiver weights did not change after AdamW step")

    # -------- STEP 7
    step(7, "generate() smoke test")
    out = model.generate([batch_item], max_new_tokens=50)
    print(f"  generated: {out!r}")
    if not (isinstance(out, list) and len(out) > 0):
        fail("generate() returned empty or non-list")

    print("\n" + "=" * 60)
    print("✅ DRY-RUN PASSED — architecture wired, ready for hour 0.")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TSLM-MD dry-run go/no-go test")
    p.add_argument(
        "--misato-h5",
        default="third_party/MiSaTo-dataset/src/inference_for_MD.hdf5",
        help="path to a small bundled MISATO HDF5 for the smoke test",
    )
    p.add_argument("--pdb-id", default="11GS", help="PDB id present in --misato-h5")
    p.add_argument("--llm-id", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--cross-attn-every-n-layers", type=int, default=1)
    p.add_argument(
        "--encoder-type", default="chronos2", choices=["chronos2", "cnn"],
        help="chronos2 = Amazon Chronos-2 encoder (recommended by OpenTSLM team); cnn = original CNNTokenizer fallback",
    )
    main(p.parse_args())
