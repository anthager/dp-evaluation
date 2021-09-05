#!/usr/bin/env bash
set -e

# the script will run with the smartnoise dir as current working directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

IMAGE=ghcr.io/anthager/smartnoise-evaluation:latest

docker run \
  -it \
  --name smartnoise-evaluation \
  --rm \
  -v $HOST_REPO_PATH/results:/dp-tools-evaluation/results \
  -v $HOST_REPO_PATH/data:/dp-tools-evaluation/data \
  -e DATASET=$DATASET \
  -e MEM_TEST="true" \
  --network smartnoise_default \
  $IMAGE \
  python src/tester.py
  
  #/bin/bash 

