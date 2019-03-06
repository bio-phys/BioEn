from __future__ import print_function
import os
import sys
import numpy as np
from bioen import optimize
from bioen import fileio as fio
import pytest


# relative tolerance for value comparison
#tol = 1.e-14
tol = 5.e-14
tol_min = 1.e-1

verbose = False
create_reference_values = False

filenames = [
    "./data/data_potra_part_2_logw_M205xN10.pkl",  # realistic test case provided by Katrin, has small theta
    "./data/data_16x15.pkl",                       # synthetic test case
    "./data/data_deer_test_logw_M808xN10.pkl",
    "./data/data_potra_part_2_logw_M205xN10.pkl",
    #    "./data/data_potra_part_1_logw_M808xN80.pkl",   ## (*)(1)
    #    "./data/data_potra_part_2_logw_M808xN10.pkl"  # ## (*)(2) with default values (tol,step_size) gsl/conj_pr gives NaN
]

def available_tests():
    exp = {}

    if (optimize.util.library_gsl()):
        exp['GSL'] = { 'bfgs2':{}  }

    if (optimize.util.library_lbfgs()):
        exp['LBFGS'] = { 'lbfgs':{} }

    return exp



def run_test_error_opt_logw(file_name=filenames[0], library='scipy/py', caching=False):
    print("=" * 80)

    if (create_reference_values):
        os.environ["OMP_NUM_THREADS"] = "1"

    if "OMP_NUM_THREADS" in os.environ:
        print("OPENMP NUM. THREADS = ", os.environ["OMP_NUM_THREADS"])

    exp = available_tests()

    # load exp. data from file
    new_mydict = fio.load_dict(file_name)
    [GInit, G, y, yTilde, YTilde, w0, theta] = fio.get_list_from_dict(new_mydict,"GInit", "G", "y", "yTilde", "YTilde", "w0", "theta")

    # run all available optimizations
    for minimizer in exp:
        for algorithm in exp[minimizer]:

            use_c_functions = True

            params = optimize.minimize.Parameters(minimizer)
            params['cache_ytilde_transposed'] = caching
            params['use_c_functions'] = use_c_functions
            params['algorithm'] = algorithm
            params['verbose'] = verbose

            if params['minimizer'] == "gsl":
                #  Force an error by defining a wrong parameter
                #params['params']['step_size'] = -1.00001
                params['algorithm'] = "TEST_INVALID"
            elif params['minimizer'] == "lbfgs":
                #  Force an error by defining a wrong parameter
                params['params']['delta'] = -1.

            if verbose:
                print("-" * 80)
                print(params)

            with pytest.raises(RuntimeError) as excinfo:
                # run optimization
                wopt, yopt, gopt, fmin_ini, fmin_fin =  \
                    optimize.log_weights.find_optimum(GInit, G, y, yTilde, YTilde, theta, params)

            print(excinfo.value)
            assert('return code' in str(excinfo.value))


def test_error_opt_logw():
    """Entry point for py.test."""
    print("")
    optimize.minimize.set_fast_openmp_flag(0)

    for file_name in filenames:
        caching = ["False"]
        run_test_error_opt_logw(file_name=file_name, caching=caching)
