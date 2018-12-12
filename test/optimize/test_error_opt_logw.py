from __future__ import print_function
import os
import sys
if sys.version_info >= (3,):
    import pickle
else:
    import cPickle as pickle
import numpy as np
from bioen import optimize
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
    exp_list_base = ""
    exp_list_gsl = [['GSL'],
                    ['bfgs2']]
    exp_list_lbfgs = [['LBFGS'],
                      ['lbfgs']]
    if (not optimize.util.library_gsl()):
        exp_list_base = exp_list_gsl

    if (optimize.util.library_lbfgs()):
        if exp_list_base != "" :
            exp_list_base = np.hstack((exp_list_base, exp_list_lbfgs))
        else:
            exp_list_base = exp_list_lbfgs
    return exp_list_base


def run_test_error_opt_logw(file_name=filenames[0], library='scipy/py', caching=False):
    print("=" * 80)

    if (create_reference_values):
        os.environ["OMP_NUM_THREADS"] = "1"

    if "OMP_NUM_THREADS" in os.environ:
        print("OPENMP NUM. THREADS = ", os.environ["OMP_NUM_THREADS"])

    exp_list = available_tests()

    # allocate vectors to store results
    exp_size = len(exp_list[0])
    yopt_list = [0] * (exp_size)
    wopt_list = [0] * (exp_size)
    gopt_list = [0] * (exp_size)
    fmin_initial_list = [0] * (exp_size)
    fmin_final_list = [0] * (exp_size)
    reeval_fmin_list = [0] * (exp_size)

    # load exp. data from file
    with open(file_name, 'r') as ifile:
        [GInit, G, y, yTilde, YTilde, w0, theta] = pickle.load(ifile)

    # run all available optimizations
    for i in range(exp_size):
        if verbose:
            print("#" * 60)

        minimizer = exp_list[0][i]
        algorithm = exp_list[1][i]

        use_c_functions = True

        params = optimize.minimize.Parameters(minimizer)
        params['cache_ytilde_transposed'] = caching
        params['use_c_functions'] = use_c_functions
        params['algorithm'] = algorithm
        params['verbose'] = verbose

        if params['minimizer'] == "gsl":
            #  Force an error by defining a wrong parameter
            params['params']['step_size'] = -1.00001
            print (params['params'])
        elif params['minimizer'] == "lbfgs":
            #  Force an error by defining a wrong parameter
            params['params']['delta'] = -1.

        if verbose:
            print("-" * 80)
            print(params)

        with pytest.raises(RuntimeError):
            # run optimization
            wopt_list[i], yopt_list[i], gopt_list[i], fmin_initial_list[i], fmin_final_list[i] =  \
                optimize.log_weights.find_optimum(GInit, G, y, yTilde, YTilde, theta, params)



def test_error_opt_logw():
    """Entry point for py.test."""
    print("")
    optimize.minimize.set_fast_openmp_flag(0)

    for file_name in filenames:
        caching = ["False"]
        run_test_error_opt_logw(file_name=file_name, caching=caching)
