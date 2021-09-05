#!/usr/bin/env bash
set -e

# the script will run with the google_dp dir as current working directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

IMAGE=ghcr.io/anthager/google-dp-evaluation:latest

docker run \
  -it \
  --name google_dp-evaluation \
  --rm \
  -v $HOST_REPO_PATH/results:/dp-tools-evaluation/results \
  -v $HOST_REPO_PATH/data:/dp-tools-evaluation/data \
  -e DATASET=$DATASET \
  -e MEM_TEST="true" \
  --network google-dp_default \
  $IMAGE \
  python src/tester.py
  
  #/bin/bash 

