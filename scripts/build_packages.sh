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


# Install system packages required by our library
#yum install -y gsl-devel
#yum install -y liblbfgs-devel

export LD_LIBRARY_PATH="/root/.local/lib"

# Compile wheels
mkdir -p wheelhouse
for PY in $CPYTHONS; do
    PYBIN=/opt/python/$PY/bin
    export BIOEN_REQUIREMENTS_TXT=$(readlink -f scripts/requirements_${PY}.txt)
    "${PYBIN}/pip3" install -r $BIOEN_REQUIREMENTS_TXT
    unset BIOEN_REQUIREMENTS_TXT
    "${PYBIN}/pip3" wheel . --no-deps -w wheelhouse
    "${PYBIN}/python3" setup.py clean
done


# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done


# We're only interested in keeping the manylinux builds
rm -vf wheelhouse/*-linux_x86_64.whl


# Check if the bundled shared objects do work, remove the primary ones
#yum remove -y gsl-devel gsl
#yum remove -y liblbfgs-devel liblbfgs
# Install packages and test
unset LD_LIBRARY_PATH
for PY in $CPYTHONS; do
    PYBIN=/opt/python/$PY/bin
    "${PYBIN}/pip3" install bioen --no-index -f ./wheelhouse
    pushd test/optimize
    "${PYBIN}/pytest" -sv
    popd
done


# simply use the lastly-set python to create the source tarball
"${PYBIN}/python3" setup.py clean
"${PYBIN}/python3" setup.py sdist --formats=gztar

