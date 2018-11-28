#!/bin/bash

set -x
set -e

# Installation location: our default is "~/.local" which would be the same as
# for the Python package when installed with
# `python setup.py install --user`
PREFIX=${HOME}/.local
mkdir -p ${PREFIX}

for DIR in gsl-2.5 liblbfgs-1.10
do
    cd ${DIR}
    ./configure --prefix=${PREFIX}
    make
    make install
    make clean
    cd -
done

echo "DONE! Successfully installed GSL and LIBLBFGS into ${PREFIX}."
