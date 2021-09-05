#!/usr/bin/env bash
ssh -T -o StrictHostKeychecking=no git@github.com

git clone git@github.com:anthager/dp-tools-evaluation.git
ln -s dp-tools-evaluation/dpevaluation /usr/local/lib/python3.6/site-packages/

/bin/bash
