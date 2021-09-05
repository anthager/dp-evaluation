#!/usr/bin/env bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)/.."

docker build \
  -t ghcr.io/anthager/tensorflow-privacy-development \
  -f development/Dockerfile \
  .
