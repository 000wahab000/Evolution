# Evolution-1 AWS Deployment Script for Windows PowerShell
param (
    [string]$AwsRegion = "us-east-1",
    [string]$ClusterName = "evolution-cluster",
    [string]$ServiceName = "evolution-svc",
    [string]$ImageName = "evolution-1",
    [string]$S3Bucket = ""
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}
if (-not (Get-Command aws -ErrorAction SilentlyContinue) -and (Test-Path "C:\Program Files\Amazon\AWSCLIV2\aws.exe")) {
    $env:Path += ";C:\Program Files\Amazon\AWSCLIV2"
}

Write-Host "=== 1. Checking AWS identity & environment ===" -ForegroundColor Cyan
try {
    $callerIdentity = aws sts get-caller-identity --output json | ConvertFrom-Json
    $accountId = $callerIdentity.Account
    Write-Host "AWS Account ID: $accountId" -ForegroundColor Green
    Write-Host "AWS Region:     $AwsRegion" -ForegroundColor Green
} catch {
    Write-Error "Failed to query AWS STS. Ensure AWS CLI is installed and configured (run 'aws configure')."
}

if (-not $S3Bucket) {
    $S3Bucket = "evolution-1-models-$accountId"
}
Write-Host "S3 Bucket:      $S3Bucket" -ForegroundColor Green

Write-Host "`n=== 2. Ensuring S3 bucket exists ===" -ForegroundColor Cyan
$bucketCheck = aws s3api head-bucket --bucket $S3Bucket 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating S3 bucket $S3Bucket..." -ForegroundColor Yellow
    if ($AwsRegion -eq "us-east-1") {
        aws s3api create-bucket --bucket $S3Bucket --region $AwsRegion
    } else {
        aws s3api create-bucket --bucket $S3Bucket --region $AwsRegion --create-bucket-configuration LocationConstraint=$AwsRegion
    }
    aws s3api put-public-access-block --bucket $S3Bucket --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
} else {
    Write-Host "S3 bucket $S3Bucket already exists." -ForegroundColor Green
}

Write-Host "`n=== 3. Ensuring ECR repository exists ===" -ForegroundColor Cyan
$ecrCheck = aws ecr describe-repositories --repository-names $ImageName --region $AwsRegion 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating ECR repository $ImageName..." -ForegroundColor Yellow
    aws ecr create-repository --repository-name $ImageName --region $AwsRegion
} else {
    Write-Host "ECR repository $ImageName already exists." -ForegroundColor Green
}

$ecrUri = "$accountId.dkr.ecr.$AwsRegion.amazonaws.com/$ImageName"

Write-Host "`n=== 4. Docker login to ECR ===" -ForegroundColor Cyan
$password = aws ecr get-login-password --region $AwsRegion
$password | docker login --username AWS --password-stdin "$accountId.dkr.ecr.$AwsRegion.amazonaws.com"

Write-Host "`n=== 5. Building & pushing Docker image ===" -ForegroundColor Cyan
docker build -t "$($ImageName):latest" .
docker tag "$($ImageName):latest" "$($ecrUri):latest"
docker push "$($ecrUri):latest"

Write-Host "`n=== 6. Registering ECS Task Definition ===" -ForegroundColor Cyan
$taskDefTemplate = Get-Content "infra/task-def.json" -Raw
$taskDefRendered = $taskDefTemplate `
    -replace '\$\{AWS_ACCOUNT_ID\}', $accountId `
    -replace '\$\{AWS_REGION\}', $AwsRegion `
    -replace '\$\{S3_BUCKET\}', $S3Bucket

$tempTaskDefPath = "infra/task-def.rendered.json"
Set-Content -Path $tempTaskDefPath -Value $taskDefRendered -Encoding UTF8

$taskDefResult = aws ecs register-task-definition --cli-input-json file://$tempTaskDefPath --output json | ConvertFrom-Json
$taskDefArn = $taskDefResult.taskDefinition.taskDefinitionArn
Write-Host "Registered Task Definition: $taskDefArn" -ForegroundColor Green
Remove-Item -Force $tempTaskDefPath -ErrorAction SilentlyContinue

Write-Host "`n=== 7. Ensuring ECS Cluster exists ===" -ForegroundColor Cyan
$clusterCheck = aws ecs describe-clusters --clusters $ClusterName --region $AwsRegion --output json | ConvertFrom-Json
$activeClusters = $clusterCheck.clusters | Where-Object { $_.status -eq "ACTIVE" }
if (-not $activeClusters) {
    Write-Host "Creating ECS cluster $ClusterName..." -ForegroundColor Yellow
    aws ecs create-cluster --cluster-name $ClusterName --region $AwsRegion
} else {
    Write-Host "ECS cluster $ClusterName already exists." -ForegroundColor Green
}

Write-Host "`n=== Deployment preparations complete! ===" -ForegroundColor Green
Write-Host "To update or run the service on Fargate:" -ForegroundColor Cyan
Write-Host "aws ecs update-service --cluster $ClusterName --service $ServiceName --task-definition $taskDefArn --force-new-deployment"
