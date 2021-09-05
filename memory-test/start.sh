#!/usr/bin/env bash
set -e

# the script will run with the smartnoise dir as current working directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

CONTAINER_IMAGE=ghcr.io/anthager/memory-test

docker run \
  --rm \
  --name memory-test \
  -v $(pwd)/../data:/dp-tools-evaluation/data \
  -v $(pwd)/../results:/dp-tools-evaluation/results \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -it \
  -e DATASET="medical-survey" \
  -e TOOL="smartnoise" \
  -e HOST_REPO_PATH="$(pwd)/.." \
  $CONTAINER_IMAGE \
  python src/run.py
