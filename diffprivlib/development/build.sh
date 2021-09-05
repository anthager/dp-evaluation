#!/usr/bin/env bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)/.."

docker build \
  -t ghcr.io/anthager/diffprivlib-development \
  -f development/Dockerfile \
  .
