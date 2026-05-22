#!/usr/bin/env bash
# Pre-clock: pull MISATO MD.hdf5 (133 GiB) + QM.hdf5 (0.3 GiB) from Zenodo
# onto an EC2 instance with a gigabit pipe, push to S3, then terminate.
#
# This turns a 3-6 hour home-Wi-Fi download into a 30-60 min unattended job.
#
# REQUIREMENTS:
#   - aws-cli configured locally (`aws configure`)
#   - An EC2 keypair you own
#   - A security group that allows SSH (port 22) from your IP
#   - An IAM role with S3 write perms attached to the instance
#       (or `aws configure` inside the instance after SSH'ing in — less ideal)
#   - About $0.30-$0.50 in AWS credits
#
# USAGE:
#   1. Edit the VARIABLES block below.
#   2. Run this script on your laptop.
#   3. SSH into the EC2 instance using the printed command.
#   4. Paste the printed inner-script and let it run unattended.
#   5. From your laptop, terminate the instance with the printed command.

set -euo pipefail

# ============ VARIABLES — EDIT THESE ============
AWS_REGION="us-east-1"
KEY_NAME="your-keypair-name"
SG_ID="sg-xxxxxxxx"
S3_BUCKET="tslm-md-data-${USER}"
INSTANCE_TYPE="c6i.xlarge"
AMI_ID="ami-0c02fb55956c7d316"   # Amazon Linux 2023 in us-east-1; change for other regions
EBS_SIZE_GB=250
IAM_INSTANCE_PROFILE=""          # optional; leave "" to skip and use aws configure on instance
# =================================================

echo "==> Creating S3 bucket s3://${S3_BUCKET} (region ${AWS_REGION})"
aws s3 mb "s3://${S3_BUCKET}" --region "${AWS_REGION}" 2>/dev/null \
  || echo "    (bucket may already exist — continuing)"

echo "==> Launching ${INSTANCE_TYPE} in ${AWS_REGION}"
RUN_ARGS=(
  --image-id "${AMI_ID}"
  --instance-type "${INSTANCE_TYPE}"
  --key-name "${KEY_NAME}"
  --security-group-ids "${SG_ID}"
  --region "${AWS_REGION}"
  --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${EBS_SIZE_GB},VolumeType=gp3}"
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=misato-fetcher}]'
)
if [ -n "${IAM_INSTANCE_PROFILE}" ]; then
  RUN_ARGS+=(--iam-instance-profile "Name=${IAM_INSTANCE_PROFILE}")
fi

INSTANCE_ID=$(aws ec2 run-instances "${RUN_ARGS[@]}" --query 'Instances[0].InstanceId' --output text)
echo "    Instance: ${INSTANCE_ID}"

echo "==> Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}" --region "${AWS_REGION}"
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "${INSTANCE_ID}" \
  --region "${AWS_REGION}" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat <<EOF

============================================================
INSTANCE READY.

SSH command:
  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}

Then PASTE the following script inside the instance:
------------------------------------------------------------
set -euo pipefail
sudo dnf install -y aria2 awscli
mkdir -p /data && cd /data

# If no IAM role: run \`aws configure\` first.

echo "Downloading MD.hdf5 (133 GiB) ..."
aria2c -x 8 -s 8 'https://zenodo.org/records/7711953/files/MD.hdf5'

echo "Downloading QM.hdf5 (0.3 GiB) ..."
aria2c -x 8 -s 8 'https://zenodo.org/records/7711953/files/QM.hdf5'

echo "Uploading to s3://${S3_BUCKET}/misato/ ..."
aws s3 cp /data/MD.hdf5 s3://${S3_BUCKET}/misato/MD.hdf5
aws s3 cp /data/QM.hdf5 s3://${S3_BUCKET}/misato/QM.hdf5

echo "DONE. Disconnect, then on your laptop run:"
echo "  aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region ${AWS_REGION}"
------------------------------------------------------------

When you're done, terminate the instance from your laptop:
  aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region ${AWS_REGION}
============================================================
EOF
