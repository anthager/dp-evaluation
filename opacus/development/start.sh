#!/usr/bin/env bash
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

export CONTAINER_NAME=opacus-dev
export IMAGE=ghcr.io/anthager/opacus-development:latest
export NETWORK=bridge # Default network

../../bin/start-dev.sh $1
