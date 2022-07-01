#!/bin/bash

# Adapted from
# https://github.com/pypa/python-manylinux-demo
# This script must have PLAT and CPYTHONS set!
#

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

# Install a system package required by our library
yum install -y atlas-devel

# Compile wheels
for PYBIN in /opt/python/{$CPYTHONS}/bin; do
    "${PYBIN}/pip" install --user -r requirements.txt
    "${PYBIN}/pip" install --user .
    "${PYBIN}/pip" wheel . --no-deps -w wheelhouse
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYBIN in /opt/python/{$CPYTHONS}/bin/; do
    "${PYBIN}/pip" install bioen --user --no-index -f ./wheelhouse
    #(cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
done

