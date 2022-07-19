#!/bin/bash

PYTHON=${PYTHON:-/opt/python/cp39-cp39/bin/python3}

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
$PYTHON setup.py sdist --formats=gztar

