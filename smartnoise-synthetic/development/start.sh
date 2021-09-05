#!/usr/bin/env bash
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

export CONTAINER_NAME=smartnoise-synthetic_development
export IMAGE=ghcr.io/anthager/smartnoise-synthetic-development:latest
# this is the default network
export NETWORK=bridge

../../bin/start-dev.sh
