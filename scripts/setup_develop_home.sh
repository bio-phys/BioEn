#!/bin/bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

export BIOEN_OPENMP=${BIOEN_OPENMP:-1}

python setup.py clean

python ./scripts/update_git_hash.py

# link all packages from ~/.local to the source location, for easy development
python setup.py config build develop --user
