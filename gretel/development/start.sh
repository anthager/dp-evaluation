#!/usr/bin/env bash
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

export CONTAINER_NAME=gretel_development
export IMAGE=ghcr.io/anthager/gretel-development:latest
# this is the default network
export NETWORK=bridge

../../bin/start-dev.sh
