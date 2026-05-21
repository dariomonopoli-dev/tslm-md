#!/usr/bin/env bash
# Deploy the trained TSLM-MD checkpoint as a SageMaker Endpoint.
#
# Justification for the pitch:
#   - Streamlit demo hits a real cloud endpoint over HTTP, not local inference
#   - Demonstrates "this is how a customer would consume the model"
#   - Multi-tenant story: one endpoint serves many computational chemistry teams
#
# Stub: implement at hour 18-20 if AWS Add #2 makes the cut.
#
# Likely shape:
#   1. Package checkpoint + tslm_md/* + inference handler into model.tar.gz
#   2. Upload model.tar.gz to s3://${S3_BUCKET}/checkpoints/
#   3. aws sagemaker create-model ...
#   4. aws sagemaker create-endpoint-config --instance-type ml.g5.xlarge ...
#   5. aws sagemaker create-endpoint --endpoint-name tslm-md-demo ...
#   6. Test with `aws sagemaker-runtime invoke-endpoint ...`
#
# Then in demo/app.py, replace local inference with:
#   import boto3
#   sm = boto3.client("sagemaker-runtime", region_name="us-east-1")
#   resp = sm.invoke_endpoint(EndpointName="tslm-md-demo", Body=...)

set -euo pipefail
echo "TODO(hour 18-20): implement SageMaker Endpoint deployment"
exit 1
