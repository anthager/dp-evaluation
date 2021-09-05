#!/usr/bin/env bash
set -e

# the script will run with the smartnoise dir as current working directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

CONTAINER_IMAGE=ghcr.io/anthager/smartnoise-synthetic-evaluation

docker run \
  --rm \
  -v $(pwd)/../../results:/dp-tools-evaluation/results \
  -it \
  -v results:/results \
  -e DATASET=medical-survey \
  -e TOOL=smartnoise_dpctgan \
  $CONTAINER_IMAGE \
  python src/generator.py

  # /bin/bash
