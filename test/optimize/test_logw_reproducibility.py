import numpy as np
from bioen import optimize
from bioen import fileio as fio


# relative tolerance for value comparison
tol = 5.e-14

filenames = [
    "./data/data_deer_test_logw_M808xN10.h5"
]


def check_logw_reproducibility(file_name, n_iter=500):
    # minimizer = 'lbfgs'
    minimizer = 'gsl'
    params = optimize.minimize.Parameters(minimizer)
    params['cache_ytilde_transposed'] = True
    params['use_c_functions'] = True
    # params['algorithm'] = "lbfgs"
    params['algorithm'] = "bfgs2"
    params['verbose'] = False

    [GInit, G, y, yTilde, YTilde, w0, theta] = fio.load(file_name,
        hdf5_keys=["GInit", "G", "y", "yTilde", "YTilde", "w0", "theta"])

    fmin_list = []
    gopt_sum_list = []
    for i in range(n_iter):
        wopt, yopt, gopt, fmin_initial, fmin_final =  \
            optimize.log_weights.find_optimum(GInit, G, y, yTilde, YTilde, theta, params)
        fmin_list.append(fmin_final)
        gopt_sum_list.append(np.sum(gopt))

    fmin_diff_list = []
    for i in range(1, n_iter):
        diff = optimize.util.compute_relative_difference_for_values(fmin_list[0], fmin_list[i])
        fmin_diff_list.append(diff)

    yopt_diff_list = []
    for i in range(1, n_iter):
        diff = optimize.util.compute_relative_difference_for_values(gopt_sum_list[0], gopt_sum_list[i])
        yopt_diff_list.append(diff)

    assert(np.sum(fmin_diff_list) < tol)
    assert(np.sum(yopt_diff_list) < tol)


def test_logw_reproducibility():
    optimize.minimize.set_fast_openmp_flag(0)
    check_logw_reproducibility(filenames[0])


optimize.minimize.set_fast_openmp_flag(0)
check_logw_reproducibility(filenames[0])
