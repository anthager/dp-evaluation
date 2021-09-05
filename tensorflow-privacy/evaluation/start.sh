#!/usr/bin/env bash
set -e

# the script will run with the smartnoise dir as current working directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

CONTAINER_IMAGE=ghcr.io/anthager/tensorflow-privacy-evaluation

docker run \
  --rm \
  --name tensorflow-privacy-evaluation \
  -v $(pwd)/../../results:/dp-tools-evaluation/results \
  -it \
  -e DATASET=medical-survey \
  $CONTAINER_IMAGE \
  python src/tester.py

  # /bin/bash
 


