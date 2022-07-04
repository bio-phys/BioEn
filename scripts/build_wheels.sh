#!/bin/bash

# Adapted from
# https://github.com/pypa/python-manylinux-demo
# This script must have the variables PLAT and CPYTHONS set, cf. gitlab-ci.yml!

set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w ./wheelhouse/
    fi
}

mkdir -p wheelhouse

# Install system packages required by our library
yum install -y gsl-devel
yum install -y liblbfgs-devel

# Compile wheels
for PY in $CPYTHONS; do
    PYBIN=/opt/python/$PY/bin
    "${PYBIN}/pip" install -r requirements.txt
    #"${PYBIN}/pip" install .
    "${PYBIN}/pip" wheel . --no-deps -w wheelhouse
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PY in $CPYTHONS; do
    PYBIN=/opt/python/$PY/bin
    "${PYBIN}/pip" install bioen --no-index -f ./wheelhouse
    #(cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
    cd test/optimize
    "${PYBIN}/pytest" -sv
    cd -
done

