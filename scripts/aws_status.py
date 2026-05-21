"""Check status of in-progress S3 multipart upload.

Avoids zsh quoting hell with long upload IDs. Run:

    python scripts/aws_status.py
    python scripts/aws_status.py --bucket my-bucket --key datasets/MD.hdf5

If your local file is at the default path (data/misato/<filename>) the script
will also print percent done, average MB/s, and ETA.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

import boto3
import botocore


def main(bucket: str, key: str, region: str, local_path: str | None) -> None:
    s3 = boto3.client("s3", region_name=region)

    # Find any in-progress multipart upload for this key.
    try:
        resp = s3.list_multipart_uploads(Bucket=bucket)
    except botocore.exceptions.ClientError as e:
        print(f"[ERROR] list_multipart_uploads failed: {e}")
        sys.exit(1)
    uploads = [u for u in (resp.get("Uploads") or []) if u["Key"] == key]

    if not uploads:
        # Maybe it's already complete.
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            print("=" * 60)
            print(f"OK  Upload COMPLETE")
            print("=" * 60)
            print(f"   key:  s3://{bucket}/{key}")
            print(f"   size: {head['ContentLength'] / 1e9:.2f} GB")
            print(f"   etag: {head['ETag']}")
            return
        except botocore.exceptions.ClientError:
            print(f"[NOT FOUND] No in-progress upload for s3://{bucket}/{key} and key does not exist.")
            return

    upload = uploads[0]
    upload_id = upload["UploadId"]
    started = upload["Initiated"]
    if not isinstance(started, _dt.datetime):
        started = _dt.datetime.fromisoformat(str(started).replace("Z", "+00:00"))

    # Paginate through parts.
    parts_total = 0
    bytes_total = 0
    paginator = s3.get_paginator("list_parts")
    for page in paginator.paginate(Bucket=bucket, Key=key, UploadId=upload_id):
        for part in page.get("Parts") or []:
            parts_total += 1
            bytes_total += part["Size"]

    print("=" * 60)
    print("..  Upload IN-PROGRESS")
    print("=" * 60)
    print(f"   key       : s3://{bucket}/{key}")
    print(f"   started   : {started.isoformat()}")
    print(f"   parts     : {parts_total}")
    print(f"   uploaded  : {bytes_total / 1e9:.2f} GB")

    # If we know the total file size, compute progress + ETA.
    if local_path is None:
        local_path = f"data/misato/{key.split('/')[-1]}"
    if os.path.exists(local_path):
        total_size = os.path.getsize(local_path)
        pct = 100 * bytes_total / max(total_size, 1)
        now = _dt.datetime.now(_dt.timezone.utc)
        elapsed = max((now - started).total_seconds(), 1)
        rate = bytes_total / elapsed
        remaining = total_size - bytes_total
        eta_sec = remaining / rate if rate > 0 else float("inf")
        print(f"   local total: {total_size / 1e9:.2f} GB")
        print(f"   percent    : {pct:.1f}%")
        print(f"   avg rate   : {rate / 1e6:.1f} MB/s")
        if eta_sec != float("inf"):
            print(f"   ETA        : {eta_sec / 60:.0f} min ({eta_sec / 3600:.1f} h)")
        else:
            print(f"   ETA        : (no progress yet)")
    else:
        print(f"   (local path {local_path} not found — skipping percent/ETA)")

    print()
    print("To ABORT this upload (frees bandwidth, deletes partial parts):")
    print(f"  aws s3api abort-multipart-upload \\")
    print(f"      --bucket {bucket} --key {key} \\")
    print(f"      --upload-id '{upload_id}' --region {region}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default="sagemaker-us-west-2-094487995066")
    p.add_argument("--key", default="datasets/MD.hdf5")
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--local-path", default=None,
                   help="path to local file to compute percent + ETA (defaults to data/misato/<filename>)")
    args = p.parse_args()
    main(args.bucket, args.key, args.region, args.local_path)
