#!/usr/bin/env bash
set -euo pipefail

# Evolution-1 AWS Deployment Script (ECS Fargate + ECR + S3)
AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="${CLUSTER_NAME:-evolution-cluster}"
SERVICE_NAME="${SERVICE_NAME:-evolution-svc}"
IMAGE_NAME="evolution-1"

echo "=== 1. Checking AWS identity & environment ==="
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account ID: ${ACCOUNT_ID}"
echo "AWS Region:     ${AWS_REGION}"

S3_BUCKET="${S3_BUCKET:-evolution-1-models-${ACCOUNT_ID}}"
echo "S3 Bucket:      ${S3_BUCKET}"

echo "=== 2. Ensuring S3 bucket exists ==="
if ! aws s3api head-bucket --bucket "${S3_BUCKET}" 2>/dev/null; then
    echo "Creating S3 bucket ${S3_BUCKET}..."
    if [ "${AWS_REGION}" = "us-east-1" ]; then
        aws s3api create-bucket --bucket "${S3_BUCKET}" --region "${AWS_REGION}"
    else
        aws s3api create-bucket --bucket "${S3_BUCKET}" --region "${AWS_REGION}" \
            --create-bucket-configuration LocationConstraint="${AWS_REGION}"
    fi
    aws s3api put-public-access-block --bucket "${S3_BUCKET}" \
        --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
else
    echo "S3 bucket ${S3_BUCKET} already exists."
fi

echo "=== 3. Ensuring ECR repository exists ==="
aws ecr describe-repositories --repository-names "${IMAGE_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${IMAGE_NAME}" --region "${AWS_REGION}"

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"

echo "=== 4. Docker login to ECR ==="
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "=== 5. Building & pushing Docker image ==="
docker build -t "${IMAGE_NAME}:latest" .
docker tag "${IMAGE_NAME}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"

echo "=== 6. Registering ECS Task Definition ==="
# Render task definition
RENDERED_TASK_DEF=$(sed \
    -e "s|\${AWS_ACCOUNT_ID}|${ACCOUNT_ID}|g" \
    -e "s|\${AWS_REGION}|${AWS_REGION}|g" \
    -e "s|\${S3_BUCKET}|${S3_BUCKET}|g" \
    infra/task-def.json)

echo "${RENDERED_TASK_DEF}" > infra/task-def.rendered.json
TASK_DEF_ARN=$(aws ecs register-task-definition --cli-input-json file://infra/task-def.rendered.json --query 'taskDefinition.taskDefinitionArn' --output text)
echo "Registered Task Definition: ${TASK_DEF_ARN}"
rm -f infra/task-def.rendered.json

echo "=== 7. Ensuring ECS Cluster exists ==="
aws ecs describe-clusters --clusters "${CLUSTER_NAME}" --region "${AWS_REGION}" --query 'clusters[?status==`ACTIVE`].clusterName' --output text | grep -w "${CLUSTER_NAME}" >/dev/null 2>&1 || \
    aws ecs create-cluster --cluster-name "${CLUSTER_NAME}" --region "${AWS_REGION}"

echo "=== Deployment preparations complete! ==="
echo "To update or run the service on Fargate:"
echo "aws ecs update-service --cluster ${CLUSTER_NAME} --service ${SERVICE_NAME} --task-definition ${TASK_DEF_ARN} --force-new-deployment"
