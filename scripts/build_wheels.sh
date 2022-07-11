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
#yum install -y gsl-devel
#yum install -y liblbfgs-devel

export LD_LIBRARY_PATH="/root/.local/lib"

# Compile wheels
for PY in $CPYTHONS; do
    PYBIN=/opt/python/$PY/bin
    export BIOEN_REQUIREMENTS_TXT=$(readlink -f scripts/requirements_${PY}.txt)
    "${PYBIN}/pip" install -r $BIOEN_REQUIREMENTS_TXT
    unset BIOEN_REQUIREMENTS_TXT
    "${PYBIN}/pip" wheel . --no-deps -w wheelhouse
done


# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

unset LD_LIBRARY_PATH

if true
then
    # Check if the bundled shared objects do work, remove the primary ones
    #yum remove -y gsl-devel gsl
    #yum remove -y liblbfgs-devel liblbfgs
    # Install packages and test
    for PY in $CPYTHONS; do
        PYBIN=/opt/python/$PY/bin
        "${PYBIN}/pip" install bioen --no-index -f ./wheelhouse
        cd test/optimize
        "${PYBIN}/pytest" -sv
        cd -
    done
fi


if true
then
    # Upload packages to local GitLab registry (in case they exist already they need to be purged first)
    ${PYBIN}/pip install typing_extensions
    ${PYBIN}/pip install twine
    #curl --request DELETE --header "PRIVATE-TOKEN: ${CI_JOB_TOKEN}" "https://gitlab.example.com/api/v4/projects/:id/packages/:package_id"
    TWINE_PASSWORD=${CI_JOB_TOKEN} TWINE_USERNAME=gitlab-ci-token \
    ${PYBIN}/python -m \
        twine upload --repository-url ${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/pypi wheelhouse/*.whl
fi
