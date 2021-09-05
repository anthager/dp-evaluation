#!/usr/bin/env bash

GREEN='\e[32m'
DEFAULT='\e[0m'
STRING='[INFO]'
INFO=${GREEN}${STRING}${DEFAULT}

echo -e "${INFO} Running SmartNoise Evaluation..."

# Run tests on dataset
python3.7 src/tester.py
