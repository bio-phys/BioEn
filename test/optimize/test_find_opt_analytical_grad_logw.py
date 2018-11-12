from __future__ import print_function
import os
import sys
if sys.version_info >= (3,):
    import pickle
else:
    import cPickle as pickle
import numpy as np
from bioen import optimize


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


# (*) require specific tunning of parameters.
# (*)(1) scipy_py/cg and scipy_c/cg
# (*)(2) GSL/conjugate_pr and GSL/bfgs


def available_tests():
    if (create_reference_values):
        # Create reference values using scipy python version for the bfgs algorithm
        exp_list_base = [['scipy_py'],
                         ['bfgs']]
    else:
        exp_list_base = [['scipy_py', 'scipy_py', 'scipy_py', 'scipy_c', 'scipy_c', 'scipy_c'],
                         ['bfgs', 'lbfgs', 'cg', 'bfgs', 'lbfgs', 'cg']]
        exp_list_gsl = [['GSL', 'GSL', 'GSL', 'GSL', 'GSL'],
                        ['conjugate_fr', 'conjugate_pr', 'bfgs2', 'bfgs', 'steepest_descent']]
        exp_list_lbfgs = [['LBFGS'],
                          ['lbfgs']]

        if (optimize.util.library_gsl()):
            exp_list_base = np.hstack((exp_list_base, exp_list_gsl))

        if (optimize.util.library_lbfgs()):
            exp_list_base = np.hstack((exp_list_base, exp_list_lbfgs))
    return exp_list_base


def run_test_optimum_logw(file_name=filenames[0], library='scipy/py', caching=False):
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
        if (minimizer == 'scipy_py'):
            minimizer = 'scipy'
            use_c_functions = False

        if (minimizer == 'scipy_c'):
            minimizer = 'scipy'

        params = optimize.minimize.Parameters(minimizer)

        params['cache_ytilde_transposed'] = caching
        params['use_c_functions'] = use_c_functions
        params['algorithm'] = algorithm
        params['verbose'] = verbose

        if verbose:
            print("-" * 80)
            print(params)

        # run optimization
        wopt_list[i], yopt_list[i], gopt_list[i], fmin_initial_list[i], fmin_final_list[i] =  \
            optimize.log_weights.find_optimum(GInit, G, y, yTilde, YTilde, theta, params)

    if (create_reference_values):
        print("-" * 80)
        print(" === CREATING REFERENCE VALUES ===")
        ref_file_name = os.path.splitext(file_name)[0] + ".ref"
        with open(ref_file_name, "wb") as f:
            pickle.dump(fmin_final_list[0], f)
        print(" [%8s][%4s] -- fmin: %.16f --> %s" % (exp_list[0][0], exp_list[1][0], fmin_final_list[0], ref_file_name))
        print("=" * 34, " END TEST ", "=" * 34)
        print("%" * 80)
    else:

        # test results

        # get reference value
        ref_file_name = os.path.splitext(file_name)[0] + ".ref"
        available_reference = False
        if (os.path.isfile(ref_file_name)):
            available_reference = True
            with open(ref_file_name, "rb") as f:
                fmin_reference = pickle.load(f)

        # print results
        print("=" * 80)
        print(" [LOGW] CHECK RESULTS for file [ %s ] -- caching(%s)" % (file_name, caching))

        if (available_reference):
            print("-" * 80)
            print(" === REFERENCE EVALUATION ===")
            for i in range(exp_size):
                eval_diff = optimize.util.compute_relative_difference_for_values(fmin_final_list[i], fmin_reference)
                print(" [%8s][%16s] -- fmin: %.16f -- fmin_reference   : %.16f  -- diff(tol=%1.0e) = %.16f" % (
                    exp_list[0][i], exp_list[1][i], fmin_final_list[i], fmin_reference, tol_min, eval_diff))

            for i in range(exp_size):
                eval_diff = optimize.util.compute_relative_difference_for_values(fmin_final_list[i], fmin_reference)
                assert(np.all(eval_diff < tol_min))

        else:
            print("-" * 80)
            print(" === REFERENCE EVALUATION === ")
            print(" [Reference not found]  To re-generate reference values enable create_reference_values=True and relaunch")

        print("-" * 80)
        print(" === RETURNED GRADIENT EVALUATION ===")
        # re-evaluation of minimum for the returned vector
        with open(file_name, 'r') as ifile:
            [GInit, G, y, yTilde, YTilde, w0, theta] = pickle.load(ifile)

        for i in range(exp_size):
            reeval_fmin_list[i] = optimize.log_weights.bioen_log_posterior(gopt_list[i], GInit, G, yTilde, YTilde, theta, use_c=True)

        # print differences
        for i in range(exp_size):
            eval_diff = optimize.util.compute_relative_difference_for_values(fmin_final_list[i], reeval_fmin_list[i])
            print(" [%8s][%16s] -- fmin: %.16f -- fmin_for_gradient: %.16f  -- diff(tol=%1.0e) = %.16f" % (
                exp_list[0][i], exp_list[1][i], fmin_final_list[i], reeval_fmin_list[i], tol, eval_diff))

        print("=" * 34, " END TEST ", "=" * 34)
        print("%" * 80)

        # validate differences
        for i in range(exp_size):
            eval_diff = optimize.util.compute_relative_difference_for_values(fmin_final_list[i], reeval_fmin_list[i])
            assert(np.all(eval_diff < tol))


def test_find_opt_analytical_grad():
    """Entry point for py.test."""
    print("")
    optimize.minimize.set_fast_openmp_flag(0)
    for file_name in filenames:
        caching_options = ["False"]
        if (not create_reference_values):
            caching_options = ["False", "True"]

        for caching in caching_options:
            run_test_optimum_logw(file_name=file_name, caching=caching)
