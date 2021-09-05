#!/usr/bin/env bash
set -e

# the script will run with the smartnoise dir as current working directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

CONTAINER_IMAGE=ghcr.io/anthager/gretel-evaluation

docker run \
  --rm \
  -v $(pwd)/../../results:/dp-tools-evaluation/results \
  -v $(pwd)/../../data:/dp-tools-evaluation/data \
  -it \
  -e DEBUG=true \
  -e DATASET=medical-survey \
  $CONTAINER_IMAGE \
  bash

  # python src/generator.py






# docker run --rm -v $(pwd)/results:/dp-tools-evaluation/results -v $(pwd)/../../data:/dp-tools-evaluation/data -it -e DATASET=medical-survey ghcr.io/anthager/gretel-evaluation python src/generator.py
