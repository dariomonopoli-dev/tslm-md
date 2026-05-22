"""Quick sanity check — invoke the deployed endpoint with one PDB.

Usage:
    python invoke_test.py --endpoint-name trajecta-tslm --pdb 1A1B --variant v1a
"""

from __future__ import annotations

import argparse
import json
import time

import boto3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", default="trajecta-tslm")
    p.add_argument("--pdb", default="1A1B")
    p.add_argument("--variant", choices=["v1a", "v1b"], default="v1a")
    p.add_argument("--region", default=None)
    p.add_argument("--mode", choices=["realtime", "async"], default="realtime")
    p.add_argument("--input-s3", default=None,
                   help="Required for async: where to upload the request JSON")
    args = p.parse_args()

    payload = json.dumps({"pdb_id": args.pdb, "variant": args.variant}).encode()

    if args.mode == "realtime":
        client = boto3.client("sagemaker-runtime", region_name=args.region)
        print(f"invoking realtime endpoint {args.endpoint_name}…")
        t0 = time.monotonic()
        resp = client.invoke_endpoint(
            EndpointName=args.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
        body = resp["Body"].read().decode()
        print(f"  responded in {(time.monotonic() - t0) * 1000:.0f} ms")
        print(json.dumps(json.loads(body), indent=2))
    else:
        if not args.input_s3:
            raise SystemExit("--input-s3 is required for async mode")
        s3 = boto3.client("s3", region_name=args.region)
        # Upload request
        bucket, *key_parts = args.input_s3.replace("s3://", "").split("/")
        key = "/".join(key_parts) + f"/req_{int(time.time())}.json"
        s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/json")
        in_s3 = f"s3://{bucket}/{key}"
        print(f"uploaded request to {in_s3}")

        client = boto3.client("sagemaker-runtime", region_name=args.region)
        out = client.invoke_endpoint_async(
            EndpointName=args.endpoint_name,
            InputLocation=in_s3,
            ContentType="application/json",
        )
        print(f"async invocation queued — output will appear at {out['OutputLocation']}")


if __name__ == "__main__":
    main()
