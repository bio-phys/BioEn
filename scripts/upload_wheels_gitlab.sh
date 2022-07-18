#!/bin/bash

# Adapted from
# https://github.com/pypa/python-manylinux-demo
# This script must have the variables PLAT and CPYTHONS set, cf. gitlab-ci.yml!

set -e -u -x

# Upload packages to local GitLab registry (in case they exist already they need to be purged first)
${PYBIN}/pip install typing_extensions
${PYBIN}/pip install twine
#curl --request DELETE --header "PRIVATE-TOKEN: ${CI_JOB_TOKEN}" "https://gitlab.example.com/api/v4/projects/:id/packages/:package_id"
TWINE_PASSWORD=${CI_JOB_TOKEN} TWINE_USERNAME=gitlab-ci-token \
${PYBIN}/python -m \
    twine upload --repository-url ${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/pypi wheelhouse/*.whl

