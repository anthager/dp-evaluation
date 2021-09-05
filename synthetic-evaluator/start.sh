#!/usr/bin/env bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"


CONTAINER_IMAGE=ghcr.io/anthager/synthetic-evaluator:latest

# docker pull $CONTAINER_IMAGE

# data dir should be in the format <sythesizer>_<epsilon>_<timestamp>

docker run \
  --rm \
  -v $(pwd)/../results:/dp-tools-evaluation/results \
  -v $(pwd)/../data:/dp-tools-evaluation/data \
  -it \
  -e DATASET=parkinson \
  $CONTAINER_IMAGE \
  python $@






# python all_statistical_results.py



# docker run --rm --name synthetic-evaluator -v $(pwd)/results:/dp-tools-evaluation/results -v $(pwd)/data:/dp-tools-evaluation/data -it -e PRIVATE_DATA=gretel_52.87_21-04-05T14:01:41 -e DATASET=turbo ghcr.io/anthager/synthetic-evaluator:latest python metadata_builder.py