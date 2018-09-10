#!/bin/bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

python setup.py build_sphinx \
|| echo "---> bioen needs to be installed first before creating the documentation!"

