#!/usr/bin/env bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)/.."

docker build \
  -t ghcr.io/anthager/opacus-development \
  -f development/Dockerfile \
  .
