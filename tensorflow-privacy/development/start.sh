#!/usr/bin/env bash
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

export CONTAINER_NAME=tensorflow-privacy-dev
export IMAGE=ghcr.io/anthager/tensorflow-privacy-development:latest
export NETWORK=bridge # Default network

../../bin/start-dev.sh $1
