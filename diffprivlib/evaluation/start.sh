#!/usr/bin/env bash
set -e

# the script will run with the smartnoise dir as current working directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

docker run \
  --rm \
  -v $(pwd)/../../results:/dp-tools-evaluation/results \
  -it \
  -e DATASET=medical-survey \
  -e SYNTH_TEST=true \
  ghcr.io/anthager/diffprivlib-evaluation:latest \
  python src/tester.py

