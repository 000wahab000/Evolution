# Evolution-1 ALB and ECS Service Provisioning Script
param (
    [string]$AwsRegion = "us-east-1",
    [string]$ClusterName = "evolution-cluster",
    [string]$ServiceName = "evolution-svc",
    [string]$AlbName = "evolution-alb",
    [string]$TargetGroupName = "evolution-tg"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}
if (-not (Get-Command aws -ErrorAction SilentlyContinue) -and (Test-Path "C:\Program Files\Amazon\AWSCLIV2\aws.exe")) {
    $env:Path += ";C:\Program Files\Amazon\AWSCLIV2"
}

Write-Host "=== Discovering Default VPC and Subnets ===" -ForegroundColor Cyan
$vpcJson = aws ec2 describe-vpcs --filters Name=isDefault,Values=true --region $AwsRegion --output json | ConvertFrom-Json
$defaultVpcId = $vpcJson.Vpcs[0].VpcId
Write-Host "VPC ID: $defaultVpcId" -ForegroundColor Green

$subnetsJson = aws ec2 describe-subnets --filters "Name=vpc-id,Values=$defaultVpcId" --region $AwsRegion --output json | ConvertFrom-Json
$subnetIds = $subnetsJson.Subnets.SubnetId
$subnetArgs = $subnetIds -join " "
Write-Host "Subnets: $($subnetIds -join ', ')" -ForegroundColor Green

Write-Host "`n=== Creating Security Groups ===" -ForegroundColor Cyan
# ALB Security Group
$albSgName = "evolution-alb-sg"
$existingAlbSg = aws ec2 describe-security-groups --filters "Name=group-name,Values=$albSgName" "Name=vpc-id,Values=$defaultVpcId" --region $AwsRegion --output json | ConvertFrom-Json
if ($existingAlbSg.SecurityGroups.Count -eq 0) {
    $albSg = aws ec2 create-security-group --group-name $albSgName --description "Security group for Evolution ALB" --vpc-id $defaultVpcId --region $AwsRegion --output json | ConvertFrom-Json
    $albSgId = $albSg.GroupId
    aws ec2 authorize-security-group-ingress --group-id $albSgId --protocol tcp --port 80 --cidr 0.0.0.0/0 --region $AwsRegion
    Write-Host "Created ALB Security Group: $albSgId" -ForegroundColor Green
} else {
    $albSgId = $existingAlbSg.SecurityGroups[0].GroupId
    Write-Host "Using existing ALB Security Group: $albSgId" -ForegroundColor Green
}

# ECS Task Security Group
$taskSgName = "evolution-task-sg"
$existingTaskSg = aws ec2 describe-security-groups --filters "Name=group-name,Values=$taskSgName" "Name=vpc-id,Values=$defaultVpcId" --region $AwsRegion --output json | ConvertFrom-Json
if ($existingTaskSg.SecurityGroups.Count -eq 0) {
    $taskSg = aws ec2 create-security-group --group-name $taskSgName --description "Security group for Evolution ECS Tasks" --vpc-id $defaultVpcId --region $AwsRegion --output json | ConvertFrom-Json
    $taskSgId = $taskSg.GroupId
    aws ec2 authorize-security-group-ingress --group-id $taskSgId --protocol tcp --port 8000 --source-group $albSgId --region $AwsRegion
    Write-Host "Created Task Security Group: $taskSgId" -ForegroundColor Green
} else {
    $taskSgId = $existingTaskSg.SecurityGroups[0].GroupId
    Write-Host "Using existing Task Security Group: $taskSgId" -ForegroundColor Green
}

Write-Host "`n=== Creating Application Load Balancer ===" -ForegroundColor Cyan
$existingAlb = aws elbv2 describe-load-balancers --names $AlbName --region $AwsRegion --output json 2>&1
if ($LASTEXITCODE -ne 0) {
    $subnetsParam = $subnetIds
    $albJson = aws elbv2 create-load-balancer --name $AlbName --subnets $subnetsParam --security-groups $albSgId --scheme internet-facing --type application --region $AwsRegion --output json | ConvertFrom-Json
    $albArn = $albJson.LoadBalancers[0].LoadBalancerArn
    $albDns = $albJson.LoadBalancers[0].DNSName
    Write-Host "Created ALB: $albDns" -ForegroundColor Green
} else {
    $albData = $existingAlb | ConvertFrom-Json
    $albArn = $albData.LoadBalancers[0].LoadBalancerArn
    $albDns = $albData.LoadBalancers[0].DNSName
    Write-Host "Using existing ALB: $albDns" -ForegroundColor Green
}

# Crucial: Set ALB idle timeout to 300s for Server-Sent Events (SSE) streaming!
Write-Host "Configuring ALB idle timeout to 300 seconds for SSE streams..." -ForegroundColor Cyan
aws elbv2 modify-load-balancer-attributes --load-balancer-arn $albArn --attributes Key=idle_timeout.timeout_seconds,Value=300 --region $AwsRegion | Out-Null

Write-Host "`n=== Creating Target Group ===" -ForegroundColor Cyan
$existingTg = aws elbv2 describe-target-groups --names $TargetGroupName --region $AwsRegion --output json 2>&1
if ($LASTEXITCODE -ne 0) {
    $tgJson = aws elbv2 create-target-group --name $TargetGroupName --protocol HTTP --port 8000 --vpc-id $defaultVpcId --target-type ip --health-check-path "/api/training/status" --health-check-interval-seconds 30 --region $AwsRegion --output json | ConvertFrom-Json
    $tgArn = $tgJson.TargetGroups[0].TargetGroupArn
    Write-Host "Created Target Group: $tgArn" -ForegroundColor Green
} else {
    $tgData = $existingTg | ConvertFrom-Json
    $tgArn = $tgData.TargetGroups[0].TargetGroupArn
    Write-Host "Using existing Target Group: $tgArn" -ForegroundColor Green
}

Write-Host "`n=== Creating ALB Listener on Port 80 ===" -ForegroundColor Cyan
$listeners = aws elbv2 describe-listeners --load-balancer-arn $albArn --region $AwsRegion --output json | ConvertFrom-Json
if ($listeners.Listeners.Count -eq 0) {
    aws elbv2 create-listener --load-balancer-arn $albArn --protocol HTTP --port 80 --default-actions Type=forward,TargetGroupArn=$tgArn --region $AwsRegion | Out-Null
    Write-Host "Created Listener on port 80 forwarding to $TargetGroupName" -ForegroundColor Green
} else {
    Write-Host "Listener already exists on ALB." -ForegroundColor Green
}

Write-Host "`n=== Ensuring ECS Service Exists ===" -ForegroundColor Cyan
$latestTaskDef = aws ecs describe-task-definition --task-definition "evolution-1" --region $AwsRegion --output json | ConvertFrom-Json
$taskDefArn = $latestTaskDef.taskDefinition.taskDefinitionArn

$serviceCheck = aws ecs describe-services --cluster $ClusterName --services $ServiceName --region $AwsRegion --output json | ConvertFrom-Json
$activeService = $serviceCheck.services | Where-Object { $_.status -eq "ACTIVE" }

$subnetListJoined = ($subnetIds | ForEach-Object { "`"$_`"" }) -join ","
$awsvpcConfig = "awsvpcConfiguration={subnets=[$subnetListJoined],securityGroups=[`"$taskSgId`"],assignPublicIp=ENABLED}"

if (-not $activeService) {
    Write-Host "Creating ECS Service $ServiceName..." -ForegroundColor Yellow
    aws ecs create-service `
        --cluster $ClusterName `
        --service-name $ServiceName `
        --task-definition $taskDefArn `
        --desired-count 1 `
        --launch-type FARGATE `
        --network-configuration "$awsvpcConfig" `
        --load-balancers "targetGroupArn=$tgArn,containerName=evolution-1,containerPort=8000" `
        --region $AwsRegion
} else {
    Write-Host "Updating ECS Service $ServiceName..." -ForegroundColor Yellow
    aws ecs update-service `
        --cluster $ClusterName `
        --service $ServiceName `
        --task-definition $taskDefArn `
        --network-configuration "$awsvpcConfig" `
        --force-new-deployment `
        --region $AwsRegion
}

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "Deployment Completed Successfully!" -ForegroundColor Green
Write-Host "Application Load Balancer URL: http://$albDns" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Green
