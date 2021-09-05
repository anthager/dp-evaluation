#!/usr/bin/env bash
set -e

# the script will run with the smartnoise dir as current working directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

mkdir -p results


docker run \
  --rm \
  -v $(pwd)/../../results:/dp-tools-evaluation/results \
  -it \
  -e DATASET=medical-survey \
  ghcr.io/anthager/opacus-evaluation:latest \
  python src/tester.py

  # python src/tester.py