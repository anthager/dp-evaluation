#!/usr/bin/env bash
set -e

# this is just a fallback script that will be run if no script is selected when running the container
echo "select a script to run, the available ones are:"
echo "$(ls)"
