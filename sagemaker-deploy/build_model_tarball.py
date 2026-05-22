"""Build model.tar.gz from training-instance artifacts.

Run this on the training g5.xlarge OR a Code Editor space that has access to
/opt/ml/checkpoints (or wherever ckpt_*.pt landed). Produces model.tar.gz
in the current directory and optionally uploads it to S3.

Usage:
    python build_model_tarball.py \\
        --v1a-ckpt /opt/ml/checkpoints/v1a/ckpt_ep1.pt \\
        --v1b-ckpt /opt/ml/checkpoints/v1b/ckpt_final.pt \\
        --preprocessed /home/sagemaker-user/preprocessed \\
        --code-dir ./code \\
        --out model.tar.gz \\
        --s3-uri s3://my-bucket/trajecta/model.tar.gz
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v1a-ckpt", type=Path, default=None,
                   help="Path to v1a checkpoint (.pt). Optional if only deploying v1b.")
    p.add_argument("--v1b-ckpt", type=Path, default=None,
                   help="Path to v1b checkpoint (.pt). Optional if only deploying v1a.")
    p.add_argument("--preprocessed", type=Path, required=True,
                   help="Directory with features_test.npz + samples_test.jsonl (MISATOMDQADataset reads from here).")
    p.add_argument("--code-dir", type=Path, default=Path(__file__).parent / "code",
                   help="Directory with inference.py + requirements.txt (default: ./code).")
    p.add_argument("--out", type=Path, default=Path("model.tar.gz"))
    p.add_argument("--s3-uri", type=str, default=None,
                   help="If set, uploads to this S3 URI after building.")
    args = p.parse_args()

    if not args.v1a_ckpt and not args.v1b_ckpt:
        raise SystemExit("provide at least one of --v1a-ckpt / --v1b-ckpt")
    if not (args.code_dir / "inference.py").exists():
        raise SystemExit(f"missing {args.code_dir/'inference.py'}")
    if not args.preprocessed.exists():
        raise SystemExit(f"--preprocessed dir does not exist: {args.preprocessed}")

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        print(f"staging in {td}")

        # Checkpoints
        if args.v1a_ckpt:
            (td / "v1a").mkdir()
            shutil.copy(args.v1a_ckpt, td / "v1a" / "ckpt_final.pt")
            print(f"  v1a ← {args.v1a_ckpt}")
        if args.v1b_ckpt:
            (td / "v1b").mkdir()
            shutil.copy(args.v1b_ckpt, td / "v1b" / "ckpt_final.pt")
            print(f"  v1b ← {args.v1b_ckpt}")

        # Preprocessed feature index — MISATOMDQADataset reads features_test.npz
        # and samples_test.jsonl from $OPENTSLM_MISATO_DATA.
        for name in ("features_test.npz", "samples_test.jsonl",
                     "norm_stats.json", "metadata.json"):
            src = args.preprocessed / name
            if src.exists():
                (td / "preprocessed").mkdir(exist_ok=True)
                shutil.copy(src, td / "preprocessed" / name)
                print(f"  preprocessed/{name}")

        # Code directory (inference.py + requirements.txt)
        shutil.copytree(args.code_dir, td / "code")
        print(f"  code/ ← {args.code_dir}")

        # Tar it up. SageMaker expects a tarball with NO leading directory.
        with tarfile.open(args.out, "w:gz") as t:
            for p in sorted(td.rglob("*")):
                if p.is_file():
                    t.add(p, arcname=str(p.relative_to(td)))

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"\nbuilt {args.out} ({size_mb:.1f} MB)")

    if args.s3_uri:
        print(f"uploading to {args.s3_uri}")
        subprocess.check_call(["aws", "s3", "cp", str(args.out), args.s3_uri])
        print("uploaded.")


if __name__ == "__main__":
    main()
