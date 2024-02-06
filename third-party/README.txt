Bundled copies of GSL and LIBLBFGS
==================================


Use the script

`install_dependencies.sh`

to conveniently install these libraries into your homedirectory.

If necessary (e.g. in case the source distribution was downloaded from PyPI),
use the script `download_dependencies.sh` to obtain the dependencies first.

As a second step, install the bioen package, e.g., using

python3 setup.py install --user

in the base directory which will detect and use the installation of GSL and LIBLBFGS.

