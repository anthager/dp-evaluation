#!/usr/bin/env bash
set -e

# the script will run with the smartnoise dir as current working directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

IMAGE=ghcr.io/anthager/smartnoise-evaluation:latest
docker-compose --project-directory ../ up -d
# docker pull $IMAGE


docker run \
  -it \
  --name smartnoise-evaluation \
  --rm \
  -v $(pwd)/../../results:/dp-tools-evaluation/results \
  -e DATASET=medical-survey \
  --network smartnoise_default \
  $IMAGE \
  python src/tester.py


  # /bin/bash 