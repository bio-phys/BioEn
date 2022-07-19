#!/bin/bash

set -e -x

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

PYBIN=${PYBIN:-/opt/python/cp39-cp39/bin}

${PYBIN}/pip3 install --no-input numpy

${PYBIN}/python3 setup.py sdist --formats=gztar

${PYBIN}/pip3 uninstall --yes numpy

