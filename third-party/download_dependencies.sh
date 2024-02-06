#!/bin/bash

set -x
set -e

curl -JLO https://ftp.gnu.org/gnu/gsl/gsl-2.5.tar.gz && tar -xzf gsl-2.5.tar.gz
curl -JLO https://github.com/downloads/chokkan/liblbfgs/liblbfgs-1.10.tar.gz && tar -xzf liblbfgs-1.10.tar.gz

echo "DONE!"
