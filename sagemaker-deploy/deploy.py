"""Create or update the Trajecta SageMaker endpoint.

Run this from a SageMaker Studio Code Editor notebook / terminal — the
SageMaker SDK picks up the execution role from the environment.

Usage:
    python deploy.py \\
        --model-data s3://my-bucket/trajecta/model.tar.gz \\
        --endpoint-name trajecta-tslm \\
        --instance-type ml.g5.xlarge \\
        --mode realtime           # or 'async'

To delete an endpoint:
    python deploy.py --endpoint-name trajecta-tslm --delete
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.async_inference import AsyncInferenceConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-data", help="S3 URI of the model.tar.gz built by build_model_tarball.py")
    p.add_argument("--endpoint-name", default="trajecta-tslm")
    p.add_argument("--instance-type", default="ml.g5.xlarge",
                   help="g5.xlarge matches the training instance; 24 GB VRAM fits both variants")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--mode", choices=["realtime", "async"], default="realtime")
    p.add_argument("--async-output-s3", default=None,
                   help="S3 prefix for async-mode output (default: sagemaker default bucket)")
    p.add_argument("--framework-version", default="2.4.0",
                   help="Torch version of the SageMaker DLC. Match the version you trained against.")
    p.add_argument("--py-version", default="py311")
    p.add_argument("--role", default=None,
                   help="IAM role ARN; defaults to sagemaker.get_execution_role()")
    p.add_argument("--region", default=None)
    p.add_argument("--hf-token", default=None,
                   help="Hugging Face token (Llama-3.2-1B is gated). "
                        "Defaults to $HUGGING_FACE_HUB_TOKEN.")
    p.add_argument("--delete", action="store_true",
                   help="Delete the endpoint + endpoint-config + model.")
    args = p.parse_args()

    region = args.region or boto3.session.Session().region_name or "us-west-2"
    sm_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    if args.delete:
        _delete(args.endpoint_name, region)
        return

    if not args.model_data:
        raise SystemExit("--model-data is required unless --delete")

    role = args.role or sagemaker.get_execution_role()
    hf_token = args.hf_token or os.getenv("HUGGING_FACE_HUB_TOKEN", "")
    if not hf_token:
        print("⚠️  HUGGING_FACE_HUB_TOKEN unset — Llama-3.2-1B is gated and the container will fail to load.", file=sys.stderr)

    env = {
        "HUGGING_FACE_HUB_TOKEN": hf_token,
        "BASE_LLM_ID": os.getenv("BASE_LLM_ID", "meta-llama/Llama-3.2-1B"),
        "LORA_R": os.getenv("LORA_R", "32"),
        "LAMBDA_REG": os.getenv("LAMBDA_REG", "0.5"),
        "TRANSFORMERS_OFFLINE": "0",
        # SageMaker PyTorch DLC quirk: tell it to install requirements.txt from code/
        "SAGEMAKER_REQUIREMENTS": "requirements.txt",
    }

    print(f"creating PyTorchModel:")
    print(f"  model_data       = {args.model_data}")
    print(f"  framework        = pytorch {args.framework_version} ({args.py_version})")
    print(f"  role             = {role}")

    model = PyTorchModel(
        model_data=args.model_data,
        role=role,
        framework_version=args.framework_version,
        py_version=args.py_version,
        entry_point="inference.py",
        source_dir=None,    # source_dir is inside model.tar.gz under code/
        env=env,
        sagemaker_session=sm_session,
        name=f"{args.endpoint_name}-model-{int(time.time())}",
    )

    deploy_kwargs = dict(
        initial_instance_count=args.instance_count,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
        wait=True,
    )

    if args.mode == "async":
        out_s3 = args.async_output_s3 or f"s3://{sm_session.default_bucket()}/trajecta/async-out/"
        print(f"  mode             = async, output {out_s3}")
        deploy_kwargs["async_inference_config"] = AsyncInferenceConfig(
            output_path=out_s3,
            max_concurrent_invocations_per_instance=4,
        )
    else:
        print(f"  mode             = realtime, instance {args.instance_type}")

    predictor = model.deploy(**deploy_kwargs)
    print(f"\n✅ endpoint up: {predictor.endpoint_name}")
    print(f"   invoke via boto3:")
    print(f"     client = boto3.client('sagemaker-runtime')")
    print(f"     client.invoke_endpoint(EndpointName='{predictor.endpoint_name}', ContentType='application/json', Body=...)")


def _delete(endpoint_name: str, region: str) -> None:
    sm = boto3.client("sagemaker", region_name=region)
    print(f"deleting endpoint {endpoint_name} in {region}")
    try:
        cfg_name = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointConfigName"]
        sm.delete_endpoint(EndpointName=endpoint_name)
        print(f"  deleted endpoint")
        try:
            model_name = sm.describe_endpoint_config(EndpointConfigName=cfg_name)["ProductionVariants"][0]["ModelName"]
            sm.delete_endpoint_config(EndpointConfigName=cfg_name)
            sm.delete_model(ModelName=model_name)
            print(f"  deleted endpoint-config {cfg_name} and model {model_name}")
        except Exception as e:
            print(f"  cleanup warning: {e}")
    except sm.exceptions.ClientError as e:
        print(f"  not found / already deleted: {e}")


if __name__ == "__main__":
    main()
