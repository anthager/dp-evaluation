#!/usr/bin/env bash
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

CONTAINER_IMAGE=ghcr.io/anthager/diffprivlib-evaluation

docker run \
  -it \
  --name diffprivlib-evaluation \
  --rm \
  -v $HOST_REPO_PATH/results:/dp-tools-evaluation/results \
  -v $HOST_REPO_PATH/data:/dp-tools-evaluation/data \
  -e DATASET=$DATASET \
  -e MEM_TEST="true" \
  $CONTAINER_IMAGE \
  python src/tester.py


  
  # /bin/bash

