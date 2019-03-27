from __future__ import print_function
import pytest
import numpy as np
from bioen import optimize
from bioen import fileio as fio

# tolerance for value comparison
tol = 5.e-14
tol_grad = 5.e-8

filename = "./data/data_deer_test_forces_M808xN10.h5"
fast_openmp_values = [0, 1]


def run_func_forces(use_c=True):
    [forces_init, w0, y, yTilde, YTilde, theta] = fio.load(filename,
        hdf5_keys=["forces_init", "w0", "y", "yTilde", "YTilde", "theta"])
    log_posterior = optimize.forces.bioen_log_posterior(forces_init, w0, y, yTilde, YTilde, theta, use_c)
    return log_posterior


def run_grad_forces(use_c=True):
    [forces_init, w0, y, yTilde, YTilde, theta] = fio.load(filename,
        hdf5_keys=["forces_init", "w0", "y", "yTilde", "YTilde", "theta"])
    fprime = optimize.forces.grad_bioen_log_posterior(forces_init, w0, y, yTilde, YTilde, theta, use_c)
    return fprime


@pytest.mark.parametrize("fast_openmp", fast_openmp_values)
def test_func_gradient_forces(fast_openmp):
    optimize.minimize.set_fast_openmp_flag(fast_openmp)
    print()
    print("fast_openmp_flag = {}".format(optimize.minimize.get_fast_openmp_flag()))
    print()

    # run GSL/C-based routines
    log_posterior_c = run_func_forces(use_c=True)
    fprime_c = run_grad_forces(use_c=True)

    # run SciPy/Python-based routines
    log_posterior_py = run_func_forces(use_c=False)
    fprime_py = run_grad_forces(use_c=False)

    df = optimize.util.compute_relative_difference_for_values(log_posterior_c, log_posterior_py)
    dg, idx = optimize.util.compute_relative_difference_for_arrays(fprime_c, fprime_py)
    print("")
    print("relative difference of function = {}".format(df))
    print("relative difference of gradient = {} (maximum at index {})".format(dg, idx))
    print("")
    assert(df < tol)
    assert(dg < tol_grad)


if __name__ == "__main__":
    test_func_gradient_forces()
