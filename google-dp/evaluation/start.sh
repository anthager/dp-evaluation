#!/usr/bin/env bash
set -e

# the script will run with the google-dp dir as current working directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

IMAGE=ghcr.io/anthager/google-dp-evaluation:latest
docker-compose --project-directory ../ up -d
# docker pull $IMAGE

docker run \
  -it \
  --name google_dp-evaluation \
  --rm \
  -v $(pwd)/../../results:/dp-tools-evaluation/results \
  -e DATASET=medical-survey \
  --network google-dp_default \
  $IMAGE \
  python src/tester.py




  # /bin/bash
