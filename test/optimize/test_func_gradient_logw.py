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
tol = 5.e-14
tol_grad = 5.e-12


def run_func(use_c=True):
    # bbfgs.use_c_bioen(use_c)
    with open("./data/data_deer_test_logw_M808xN10.pkl", 'r') as ifile:
        [GInit, G, y, yTilde, YTilde, w0, theta] = pickle.load(ifile)
    g = GInit.copy()
    gPrime = np.asarray(g[:].T)[0]
    log_posterior = optimize.log_weights.bioen_log_posterior(gPrime, g, G, yTilde, YTilde, theta, use_c=use_c)
    return log_posterior


def run_grad(use_c=True):
    # bbfgs.use_c_bioen(use_c)
    with open("./data/data_deer_test_logw_M808xN10.pkl", 'r') as ifile:
        [GInit, G, y, yTilde, YTilde, w0, theta] = pickle.load(ifile)
    g = GInit.copy()
    gPrime = np.asarray(g[:].T)[0]
    fprime = optimize.log_weights.grad_bioen_log_posterior(gPrime, g, G, yTilde, YTilde, theta, use_c=use_c)
    return fprime


def test_func_gradient():
    optimize.minimize.set_fast_openmp_flag(1)
    print()
    print("fast_openmp_flag = {}".format(optimize.minimize.get_fast_openmp_flag()))
    print()

    # run C-based routines
    log_posterior_c = run_func(use_c=True)
    fprime_c = run_grad(use_c=True)

    # run Python-based routines
    log_posterior_py = run_func(use_c=False)
    fprime_py = run_grad(use_c=False)

    # print ""
    # print "Min.C :", log_posterior_c
    # print "Min.Py:",log_posterior_py

    df = optimize.util.compute_relative_difference_for_values(log_posterior_c, log_posterior_py)
    dg, idx = optimize.util.compute_relative_difference_for_arrays(fprime_c, fprime_py)
    print("")
    print("relative difference of function = {}".format(df))
    print("relative difference of gradient = {} (maximum at index {})".format(dg, idx))
    print("")
    assert(df < tol)
    assert(dg < tol_grad)


if __name__ == "__main__":
    test_func_gradient()
