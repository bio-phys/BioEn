#!/bin/bash

set -x
set -e

# Installation location:
# * The default is "~/.local" which would be the same as for the Python package
#   when installed with `python setup.py install --user`
# * To specify another location, set PREFIX in the environment, e.g. by running
#   PREFIX=/some/path ./install_dependencies.sh
PREFIX=${PREFIX:=${HOME}/.local}
mkdir -p ${PREFIX}

for DIR in gsl-2.5 liblbfgs-1.10
do
    cd ${DIR}
    ./configure --prefix=${PREFIX}
    make -j4
    make install
    make clean
    cd -
done

echo "DONE! Successfully installed GSL and LIBLBFGS into ${PREFIX}."
