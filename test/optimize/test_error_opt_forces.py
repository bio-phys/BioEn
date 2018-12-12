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
    "./data/data_deer_test_forces_M808xN10.pkl",
    "./data/data_forces_M64xN64.pkl"
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


def run_test_error_forces(file_name=filenames[0], caching=False):

    print("=" * 80)

    if (create_reference_values):
        os.environ["OMP_NUM_THREADS"] = "1"

    if "OMP_NUM_THREADS" in os.environ:
        print(" OPENMP NUM. THREADS = ", os.environ["OMP_NUM_THREADS"])

    exp_list = available_tests()

    # if GSL and LBFGS are not active we do not test error handling.
    if exp_list != "" :
        # allocate vectors to store results
        exp_size = len(exp_list[0])
        wopt_list = [0] * (exp_size)
        yopt_list = [0] * (exp_size)
        forces_list = [0] * (exp_size)
        fmin_initial_list = [0] * (exp_size)
        fmin_final_list = [0] * (exp_size)
        chiSqr = [0] * (exp_size)
        S = [0] * (exp_size)
        reeval_fmin_list = [0] * (exp_size)

        # load exp. data from file
        with open(file_name, 'rb') as ifile:
            [forces_init, w0, y, yTilde, YTilde, theta] = pickle.load(ifile)

        # run all available optimizations
        for i in range(exp_size):

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
                print("", params)

            with pytest.raises(RuntimeError):
                wopt_list[i], yopt_list[i], forces_list[i], fmin_initial_list[i], fmin_final_list[i], chiSqr[i], S[i] = \
                    optimize.forces.find_optimum(forces_init, w0, y, yTilde, YTilde, theta, params)


def test_error_opt_forces():
    """Entry point for py.test."""
    print("")
    optimize.minimize.set_fast_openmp_flag(0)
    for file_name in filenames:
        caching = ["False"]
        run_test_error_forces(file_name=file_name, caching=caching)
