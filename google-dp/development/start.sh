#!/usr/bin/env bash
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

# Start PostgreSQL container
docker-compose up -d

export CONTAINER_NAME=google-dp-dev
export IMAGE=ghcr.io/anthager/google-dp-development:latest
export NETWORK=google-dp_default
# Start google-dp container
../../bin/start-dev.sh
