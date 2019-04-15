import numpy as np
from bioen import optimize
from bioen import fileio as fio


# relative tolerance for value comparison
tol = 5.e-14

filenames = [
    "./data/data_deer_test_forces_M808xN10.h5"
]


def check_forces_reproducibility(file_name, n_iter=500):
    # minimizer = 'lbfgs'
    minimizer = 'gsl'
    params = optimize.minimize.Parameters(minimizer)
    params['cache_ytilde_transposed'] = True
    params['use_c_functions'] = True
    # params['algorithm'] = "lbfgs"
    params['algorithm'] = "bfgs2"
    params['verbose'] = False

    [forces_init, w0, y, yTilde, YTilde, theta] = fio.load(file_name,
        hdf5_keys=["forces_init", "w0", "y", "yTilde", "YTilde", "theta"])

    fmin_list = []
    forces_sum_list = []
    for i in range(n_iter):
        wopt, yopt, forces, fmin_initial, fmin_final, chiSqr, S = \
            optimize.forces.find_optimum(forces_init, w0, y, yTilde, YTilde, theta, params)
        fmin_list.append(fmin_final)
        forces_sum_list.append(np.sum(forces))

    fmin_diff_list = []
    for i in range(1, n_iter):
        diff = optimize.util.compute_relative_difference_for_values(fmin_list[0], fmin_list[i])
        fmin_diff_list.append(diff)

    forces_diff_list = []
    for i in range(1, n_iter):
        diff = optimize.util.compute_relative_difference_for_values(forces_sum_list[0], forces_sum_list[i])
        forces_diff_list.append(diff)

    assert(np.sum(fmin_diff_list) < tol)
    assert(np.sum(forces_diff_list) < tol)


def test_forces_reproducibility():
    optimize.minimize.set_fast_openmp_flag(0)
    check_forces_reproducibility(filenames[0])
