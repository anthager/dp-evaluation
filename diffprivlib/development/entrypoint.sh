#!/usr/bin/env bash
ssh -T -o StrictHostKeychecking=no git@github.com

git clone git@github.com:anthager/dp-tools-evaluation.git
# Link dpevaluation package to default python library to allow import
ln -s /dp-tools-evaluation/dpevaluation /usr/local/lib/python3.7/site-packages/

/bin/bash
