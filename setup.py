import os
import sys
import glob
import numpy
import platform
import subprocess as sub
from setuptools import setup, Command, Extension
from Cython.Distutils import build_ext


def get_version():
    """Get the version number from version.py in the source tree."""
    try:
        pkg_dir = os.path.abspath("bioen")
        sys.path.insert(0, pkg_dir)
        from bioen.version import get_version_string
        sys.path.pop(0)
        ver = get_version_string()
    except:
        ver = "n/a"
    return ver


def on_mac():
    """Check if we're running on a Mac."""
    if "Darwin" in platform.system():
        return True
    else:
        return False


# query intel or gcc (default) compiler
def use_icc():
    """Check if the Intel compiler shall be used."""
    q = 0
    try:
        CC = os.environ["CC"]
        if CC == "icc":
            q = 1
    except:
        q = 0
    return q


def get_gcc_ver(gcc="gcc"):
    """Determine the version of GCC. Returns a tuple with integers."""
    cmd = [gcc, '-v']
    major = -1
    minor = -1
    patch = -1
    raw = sub.check_output(cmd, stderr=sub.STDOUT).decode('ascii').lower().split('\n')
    for line in raw:
        if line.startswith('gcc version'):
            tokens = line.split()
            # we obtain a version string such as "5.4.0"
            verstr = tokens[2].strip()
            vertup = verstr.split('.')
            major = int(vertup[0])
            minor = int(vertup[1])
            patch = int(vertup[2])
    ver = major, minor, patch
    return ver


# query boolean environment variable for OpenMP
try:
    BIOEN_OPENMP = bool(int(os.environ["BIOEN_OPENMP"]))
except:
    BIOEN_OPENMP = False


# query environment variable to set external CFLAGS
try:
    BIOEN_CFLAGS = os.environ["BIOEN_CFLAGS"].split()
except:
    BIOEN_CFLAGS = False


if not BIOEN_CFLAGS:
    # query boolean environment variable whether to use fast compile flags
    try:
        BIOEN_FAST_CFLAGS = bool(int(os.environ["BIOEN_FAST_CFLAGS"]))
    except:
        BIOEN_FAST_CFLAGS = True


# Query if liblbfgs shall be searched for at the system default location.
# This flag is only relevent if LBFGS_HOME is not set in the environment.
try:
    BIOEN_USE_DEFAULT_LBFGS = bool(int(os.environ["BIOEN_USE_DEFAULT_LBFGS"]))
except:
    BIOEN_USE_DEFAULT_LBFGS = False


# Query if gsl shall be searched for at the system default location.
# This flag is only relevent if GSL_HOME is not set in the environment.
try:
    BIOEN_USE_DEFAULT_GSL = bool(int(os.environ["BIOEN_USE_DEFAULT_GSL"]))
except:
    BIOEN_USE_DEFAULT_GSL = True


# influence setting the rpath
try:
    BIOEN_RPATH = bool(os.environ["BIOEN_RPATH"])
except:
    BIOEN_RPATH = True


# compile the Cython and C extension
SRC = []
SRC.append("bioen/optimize/ext/c_bioen.pyx")
SRC.append("bioen/optimize/ext/c_bioen_common.c")
SRC.append("bioen/optimize/ext/c_bioen_kernels_logw.c")
SRC.append("bioen/optimize/ext/c_bioen_kernels_forces.c")
SRC.append("bioen/optimize/ext/c_bioen_error.c")

INCLUDE_DIRS = []
INCLUDE_DIRS.append(numpy.get_include())


# build-up compile flags, handling less and more aggressive options
CFLAGS = []
# CFLAGS.append("-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")
if BIOEN_CFLAGS:
    CFLAGS = BIOEN_CFLAGS
else:
    if BIOEN_FAST_CFLAGS:
        CFLAGS.append("-O3")
    else:
        CFLAGS.append("-O2")
    CFLAGS.append("-fPIC")
    CFLAGS.append("-Wno-unknown-pragmas")
    if not on_mac():
        if BIOEN_FAST_CFLAGS:
            CFLAGS.append("-march=native")
            # --- do not use 'fast-math', it causes wrong results ---
            # CFLAGS.append("-ffast-math")
            # CFLAGS.append("-fopt-info")
        else:
            CFLAGS.append("-msse4.2")
            CFLAGS.append("-mtune=native")
        CFLAGS.append("-std=gnu99")


if BIOEN_OPENMP:
    if use_icc():
        CFLAGS.append("-qopenmp")
        CFLAGS.append("-DUSE_OMP_VECTORS")
        CFLAGS.append("-DUSE_OMP_THREADS")
    else:
        CFLAGS.append("-fopenmp")
        gcc = get_gcc_ver()
        if gcc[0] >= 5:
            CFLAGS.append("-DUSE_OMP_VECTORS")
        CFLAGS.append("-DUSE_OMP_THREADS")


LDFLAGS = []
LDFLAGS.append("-lm")
if not on_mac():
    if BIOEN_OPENMP:
        if use_icc():
            LDFLAGS.append("-shared")
            LDFLAGS.append("-lsvml")
            LDFLAGS.append("-qopenmp")
        else:
            LDFLAGS.append("-fopenmp")
            LDFLAGS.append("-lgomp")


# check if the libraries are already installed into "~/.local" using `install_dependencies.sh`
THIRD_PARTY_PREFIX = os.path.join(os.environ['HOME'], ".local")

if 'LBFGS_HOME' in os.environ:
    LBFGS = os.environ['LBFGS_HOME']
    CFLAGS.append("-DENABLE_LBFGS")
    INCLUDE_DIRS.append(os.path.join(LBFGS, "include"))
    LDFLAGS.append("-L" + os.path.join(LBFGS, "lib"))
    if BIOEN_RPATH:
        LDFLAGS.append("-Wl,-rpath," + os.path.join(LBFGS, "lib"))
    LDFLAGS.append("-llbfgs")
elif os.path.isfile(os.path.join(THIRD_PARTY_PREFIX, "include/lbfgs.h")):
    LBFGS = THIRD_PARTY_PREFIX
    CFLAGS.append("-DENABLE_LBFGS")
    INCLUDE_DIRS.append(os.path.join(LBFGS, "include"))
    LDFLAGS.append("-L" + os.path.join(LBFGS, "lib"))
    if BIOEN_RPATH:
        LDFLAGS.append("-Wl,-rpath," + os.path.join(LBFGS, "lib"))
    LDFLAGS.append("-llbfgs")
else:
    if BIOEN_USE_DEFAULT_LBFGS:
        CFLAGS.append("-DENABLE_LBFGS")
        LDFLAGS.append("-llbfgs")
    else:
        print("Warning: liblbfgs is not used!")


if 'GSL_HOME' in os.environ:
    GSL = os.environ['GSL_HOME']
    CFLAGS.append("-DENABLE_GSL")
    INCLUDE_DIRS.append(os.path.join(GSL, "include"))
    LDFLAGS.append("-L" + os.path.join(GSL, "lib"))
    LDFLAGS.append("-Wl,-rpath," + os.path.join(GSL, "lib"))
    LDFLAGS.append("-lgsl")
    LDFLAGS.append("-lgslcblas")
elif os.path.isfile(os.path.join(THIRD_PARTY_PREFIX, "include/gsl/gsl_vector.h")):
    GSL = THIRD_PARTY_PREFIX
    CFLAGS.append("-DENABLE_GSL")
    INCLUDE_DIRS.append(os.path.join(GSL, "include"))
    LDFLAGS.append("-L" + os.path.join(GSL, "lib"))
    LDFLAGS.append("-Wl,-rpath," + os.path.join(GSL, "lib"))
    LDFLAGS.append("-lgsl")
    LDFLAGS.append("-lgslcblas")
else:
    if BIOEN_USE_DEFAULT_GSL:
        CFLAGS.append("-DENABLE_GSL")
        LDFLAGS.append("-lgsl")
        LDFLAGS.append("-lgslcblas")
    else:
        print("Warning: GSL is not used!")


ver = get_version()
print("---")
print("bioen v" + ver)
print("BIOEN_OPENMP : " + str(BIOEN_OPENMP))
if not BIOEN_CFLAGS:
    print("BIOEN_FAST_CFLAGS : " + str(BIOEN_FAST_CFLAGS))
else:
    print("Note: user-defined CFLAGS")
print("CFLAGS : " + str(CFLAGS))
print("LDFLAGS : " + str(LDFLAGS))
print("RPATH : " + str(BIOEN_RPATH))
print("---")


ext = []
ext.append(
    Extension("bioen.optimize.ext.c_bioen",
              SRC,
              include_dirs=INCLUDE_DIRS,
              extra_compile_args=CFLAGS,
              extra_link_args=LDFLAGS
              )
)

# building the wheels may require older versions of the reqs
try:
    BIOEN_REQUIREMENTS_TXT = os.environ["BIOEN_REQUIREMENTS_TXT"]
except:
    BIOEN_REQUIREMENTS_TXT = "requirements.txt"
reqs = [l.strip() for l in open(BIOEN_REQUIREMENTS_TXT).readlines()]

class CleanCommand(Command):
    """Custom clean command to remove unnecessary files."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        rm_args = ['build', 'doc/html', 'doc/doctrees', 'bioen.egg-info',
                   'bioen/githash.py', 'bioen/optimize/ext/c_bioen.c']
        os.system("rm -vrf " + " ".join(rm_args))
        os.system("find . -name '*.o' -delete -print")
        os.system("find . -name '*.so' -delete -print")
        os.system("find . -name '*.pyc' -delete -print")
        os.system("find . -type d -name __pycache__ | while read DIR ; do rm -vrf ${DIR} ; done")


setup(
    name='bioen',
    version=ver,
    description='Bayesian Inference Of ENsembles',
    author='Katrin Reichel, Juergen Koefinger, Cesar Allande, Klaus Reuter, Lukas S. Stelzl',
    author_email='bioen@biophys.mpg.de',
    packages=["bioen",
              "bioen.analyze",
              "bioen.analyze.observables",
              "bioen.analyze.observables.generic",
              "bioen.analyze.observables.cd_data",
              "bioen.analyze.observables.deer",
              "bioen.analyze.observables.scattering",
              "bioen.analyze.show_plot",
              "bioen.optimize",
              "bioen.optimize.ext"],
    package_data={'bioen': ['bioen/optimize/config/bioen_optimize.yaml']},
    cmdclass={'clean': CleanCommand, 'build_ext': build_ext},
    ext_modules=ext,
    include_package_data=True,
    install_requires=reqs,
    entry_points={'console_scripts':
                  ['bioen = bioen.analyze.run_bioen:main']},
    zip_safe=False,
)
