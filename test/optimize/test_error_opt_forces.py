"""Test if the forces minimizer throws an error when invalid parameters are passed.
"""

from __future__ import print_function
import os
import numpy as np
from bioen import optimize
from bioen import fileio as fio
import pytest


filenames = [
    "./data/data_deer_test_forces_M808xN10.h5",
    "./data/data_forces_M64xN64.h5"
]


def available_tests():
    exp = {}

    if (optimize.util.library_gsl()):
        exp['GSL'] = {'bfgs2': {}}

    if (optimize.util.library_lbfgs()):
        exp['LBFGS'] = {'lbfgs': {}}

    return exp


def run_test_error_forces(file_name=filenames[0], caching=False):

    print("=" * 80)

    if "OMP_NUM_THREADS" in os.environ:
        print(" OPENMP NUM. THREADS = ", os.environ["OMP_NUM_THREADS"])

    exp = available_tests()

    [forces_init, w0, y, yTilde, YTilde, theta] = fio.load(file_name,
        hdf5_keys=["forces_init", "w0", "y", "yTilde", "YTilde", "theta"])

    for minimizer in exp:
        for algorithm in exp[minimizer]:

            params = optimize.minimize.Parameters(minimizer)
            params['cache_ytilde_transposed'] = caching
            params['use_c_functions'] = True
            params['algorithm'] = algorithm
            params['verbose'] = True

            if params['minimizer'] == "gsl":
                #  Force an error by defining a wrong parameter
                params['algorithm'] = "TEST_INVALID"
                #params['params']['step_size'] = -1.00001

                # print (params['params'])
            elif params['minimizer'] == "lbfgs":
                #  Force an error by defining a wrong parameter
                params['params']['delta'] = -1.

            print("-" * 80)
            print("", params)

            with pytest.raises(RuntimeError) as excinfo:
                wopt, yopt, forces, fmin_ini, fmin_fin, chiSqr, S = \
                    optimize.forces.find_optimum(forces_init, w0, y, yTilde, YTilde, theta, params)

            print(excinfo.value)
            assert('return code' in str(excinfo.value))


def test_error_opt_forces():
    """Entry point for py.test."""
    print("")
    optimize.minimize.set_fast_openmp_flag(0)
    for file_name in filenames:
        caching = ["False"]
        run_test_error_forces(file_name=file_name, caching=caching)
