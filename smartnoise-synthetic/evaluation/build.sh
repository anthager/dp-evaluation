#!/usr/bin/env bash
set -e
# docker on linux cant set context to the dir above cwd so we go one level up here
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)/../.."

CONTAINER_IMAGE=ghcr.io/anthager/smartnoise-synthetic-evaluation

docker build \
  -t $CONTAINER_IMAGE \
  -f smartnoise-synthetic/evaluation/Dockerfile \
  .
