#!/usr/bin/env bash
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

# Start PostgreSQL container
docker-compose up -d

export CONTAINER_NAME=smartnoise-dev
export IMAGE=ghcr.io/anthager/smartnoise-development:latest
export NETWORK=smartnoise_default
# Start smartnoise docker container
../../bin/start-dev.sh
