#!/usr/bin/env bash
set -e
# docker on linux cant set context to the dir above cwd so we go one level up here
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)/../.."

docker build \
  -t ghcr.io/anthager/google-dp-evaluation \
  -f google-dp/evaluation/Dockerfile \
  .
