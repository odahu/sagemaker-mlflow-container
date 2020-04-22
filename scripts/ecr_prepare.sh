#!/usr/bin/env bash

# The name of our algorithm
repo_name=$1

if [[ -z "$1" ]]
then
  echo "Pass repo name (ecr_prepare.sh [REPO_NAME])"
  exit 255
fi


account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)

registry="${account}.dkr.ecr.${region}.amazonaws.com"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${repo_name}" > /dev/null 2>&1

if [[ $? -ne 0 ]]
then
    aws ecr create-repository --repository-name "${repo_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${registry} > /dev/null

echo ${registry}