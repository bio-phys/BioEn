#!/bin/bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
python setup.py sdist --formats=gztar
