"""Quick HDF5 introspection. Avoids zsh/heredoc quoting pitfalls.

Usage:
    python scripts/inspect_h5.py <path> [--key <name>]
"""

from __future__ import annotations

import argparse
import h5py


def main(args: argparse.Namespace) -> None:
    with h5py.File(args.path, "r") as f:
        keys = list(f.keys())
        print(f"file              : {args.path}")
        print(f"top-level entries : {len(keys)}")
        print(f"first 5 keys      : {keys[:5]}")
        target = args.key or keys[0]
        if target not in f:
            print(f"\nrequested key {target!r} not found.")
            return
        print(f"\n=== contents of '{target}' ===")
        obj = f[target]
        if hasattr(obj, "keys"):
            for k in obj.keys():
                child = obj[k]
                if hasattr(child, "shape"):
                    print(f"  {k}: shape={child.shape}, dtype={child.dtype}")
                else:
                    print(f"  {k}: subgroup ({len(list(child.keys()))} children)")
        else:
            print(f"  is a dataset: shape={obj.shape}, dtype={obj.dtype}")
            try:
                vals = obj[...] if obj.size < 20 else f"{obj[:5]} ..."
                print(f"  values: {vals}")
            except Exception as e:
                print(f"  (can't read: {e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--key", default=None, help="specific group/dataset to inspect")
    main(p.parse_args())
