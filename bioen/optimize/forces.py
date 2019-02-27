"""Python interface for the forces method
"""
from __future__ import print_function

import numpy as np
import scipy.optimize as sopt

import os
import sys
import time
import struct

from . import common
from .ext import c_bioen
from . import minimize


# FORCES Synthetic data generation functions
def gen_synthetic_data(M, N, YTrue, sig_exp):
    """
    Observed values drawn from normal distribution.

    Parameters
    ----------
    M: double
    N: double
    YTrue:
    sig_exp:

    Returns
    -------
    YObs
    YTilde
    """
    # np.random.seed(0)
    YObs = np.random.normal(YTrue, sig_exp)
    YObs = np.array(YObs)
    # Experimentally observed values scaled by the inverse of the experimental error
    YTilde = YObs / sig_exp
    return YObs, YTilde


def gen_sythetic_ensemble(M, N, YTrue, sig_exp, sig_sim):
    """
    Parameters
    ----------
    M: double
    N: double
    YTrue:
    sig_exp:
    sig_sim:

    Returns
    -------
    y:
    yTilde:

    """
    # Values of observables for simulation ensemble.
    # Also distributed around true values, but with different standard deviation.
    # np.random.seed(0)
    y = np.array(np.random.normal(np.repeat(np.transpose([YTrue]), N, axis=1), sig_sim))
    # Values of ensemble observables scaled by the inverse of the experimental error
    yTilde = y.copy()
    sig_exp_mat = np.transpose(np.repeat([sig_exp], N, axis=0))
    yTilde /= sig_exp_mat
    # initial weights
    return y, yTilde


def init_forces(M, val=0):
    """
    Parameters
    ----------
    M:
    val:

    Returns
    -------
    forces:

    """
    forces = np.array(np.zeros((M, 1)))
    forces[:, 0] = val
    return forces


# FORCES general functions


def get_weights_from_forces(w0, y, forces):
    """
    If forces are zero then w is equal to w0.
    Parameters
    ----------
    w0: Nx1 matrix
    forces: 1xM matrix
    yTilde: MxN matrix
    Returns
    -------
    w: array of length N
    """

    if (forces.ndim == 1):
        forces = np.expand_dims(forces, axis=0)

    x = np.dot(forces, y)
    M = np.exp(x - np.max(x))
    w = np.asarray(w0) * np.asarray(M.T)
    result = w / w.sum()
    return result

 


def bioen_chi2_s_forces(forces, w0, yTilde, YTilde):
    """
    Parameters
    ----------
    forcesInit: 1xM matrix
    w0: array of length N
    yTilde: MxN matrix, M observables y_i / sigma_i for the M structures
    YTilde: 1xM matrix, experimental observables

    Returns
    -------
    forces:
    """

    forces = forces.T
    w = get_weights_from_forces(w0, yTilde, forces)

    chiSqr = common.chiSqrTerm(w, yTilde, YTilde)
    # selecting non-zero weights because lim w->0 w log(w) = 0
    ind = np.where(w > 0)[0]
    return np.dot((np.log(w[ind] / w0[ind])).T, w[ind])[0, 0], chiSqr


def check_params_forces(forcesInit, w0, y, yTilde, YTilde):
    """
    Check the proper dimensionality of the forces parameters.
    The reference dimensions are based on the (m, n) matrix yTilde

    Parameters
    ----------
    forcesInit : ndarray (m, 1)
    w0 : ndarray (n, 1)
    y : ndarray (m, n)
        m observables calculate for the n structures
    yTilde : ndarray (m, n)
        m observables y_i / sigma_i for the n structures
    YTilde : ndarray (1, m)
        experimental observables

    Raises
    ------
    ValueError
    """
    m, n = yTilde.shape
    error = False

    def _pretty_print(name, expected, current):
        print("Unexpected shape for variable: {name}\n"
              "Expected: {expected}\n"
              "Current:  {current}".format(name=name, expected=expected, current=current))

    # check for correct dimensions
    if forcesInit.shape != (m, 1):
        _pretty_print("forcesInit", (m, 1), forcesInit.shape)
        error = True
    if w0.shape != (n, 1):
        _pretty_print('w0', (n, 1), w0.shape)
        error = True
    if y.shape != (m, n):
        _pretty_print('y', (m, n), y.shape)
        error = True
    if YTilde.shape != (1, m):
        _pretty_print('YTilde', (1, m), YTilde.shape)
        error = True

    if error:
        raise ValueError("arguments dimensionality for the 'forces' method are wrong")


# FORCES function insterace selector
# TODO: y parameter should be removed (only used on the optimizer)
def bioen_log_posterior(forces, w0, y, yTilde, YTilde, theta, use_c=True, caching=False):
    """
    Interface to select between C+GSL and Python versions for the objective
    function using the forces method

    Parameters
    ----------
    forcesInit: 1xM matrix
    w0: array of length N
    yTilde: MxN matrix, M observables y_i / sigma_i for the M structures
    YTilde: 1xM matrix, experimental observables
    theta: float, confidence parameter,

    Returns
    -------
    double:
      fmin value
    """
    #print ("BIOEN_LOG_POSTERIOR INTERFACE")
    if use_c:
        log_posterior = c_bioen.bioen_log_posterior_forces(forces, w0, yTilde, YTilde, theta)
    else:
        log_posterior = bioen_log_posterior_base(forces, w0, yTilde, YTilde, theta)

    return log_posterior


# FORCES gradient interface selector
# TODO: y parameter should be removed (only used on the optimizer)
def grad_bioen_log_posterior(forces, w0, y, yTilde, YTilde, theta, use_c=True, caching=False):
    """
    Interface to select between C+GSL and Python versions for the gradient
    function using the forces method

    Parameters
    ----------
    forces: 1xM matrix
    w0: array of length N
    y: MxN matrix, M observables calculate for the M structures    #### unused
    yTilde: MxN matrix, M observables y_i / sigma_i for the M structures
    YTilde: 1xM matrix, experimental observables
    theta: float, confidence parameter,

    Returns
    -------
    fprime: 1xN matrix
    """

    #print ("GRAD_BIOEN_LOG_POSTERIOR INTERFACE")
    if use_c:
        fprime = c_bioen.grad_bioen_log_posterior_forces(forces, w0, yTilde, YTilde, theta)
    else:
    
        fprime = grad_bioen_log_posterior_base(forces, w0, yTilde, YTilde, theta)

    return fprime


# FORCES function
def bioen_log_posterior_base(forces, w0, yTilde, YTilde, theta, use_c=True):
    """
    Legacy Python version of bioen_log_posterior forces method

    Parameters
    ----------
    forcesInit: 1xM matrix
    w0: array of length N
    yTilde: MxN matrix, M observables y_i / sigma_i for the M structures
    YTilde: 1xM matrix, experimental observables
    theta: float, confidence parameter

    Returns
    -------
    double:
      fmin value
    """
    #print ("BIOEN_LOG_POSTERIOR -- needs to be fixed")
    #print ("    forces shape", forces.shape)
    #print ("    w0 shape    ", w0.shape)
    #print ("    yTilde shape", yTilde.shape)
    #print ("    YTilde shape", YTilde.shape)
    #print ("    forces type ", str(type(forces)))
    #print ("    w0 type     ", str(type(w0)))
    #print ("    yTilde type ", str(type(yTilde)))
    #print ("    YTilde type ", str(type(YTilde)))


    forces = forces.T
    w = get_weights_from_forces(w0, yTilde, forces)

    #print ("    w shape", w.shape)
    #print ("    w type ", str(type(YTilde)))
    chiSqr = common.chiSqrTerm(w, yTilde, YTilde)
    # selecting non-zero weights because lim_{w->0} w log(w) = 0
    ind = np.where(w > 0)[0]

    result =  theta * np.dot((np.log(w[ind] / w0[ind])).T, w[ind])[0, 0] + chiSqr

    #print ("current fmin is" , result)

    return result
    #return theta * np.dot((np.log(w[ind] / w0[ind])).T, w[ind])[0, 0] + chiSqr


# FORCES gradient
def grad_bioen_log_posterior_base(forces, w0, yTilde, YTilde, theta, use_c=True):
    """
    Legacy Python version of grad_bioen_log_posterior forces method

    Parameters
    ----------
    forces: 1xM matrix
    w0: array of length N
    yTilde: MxN matrix, M observables y_i / sigma_i for the M structures
    YTilde: 1xM matrix, experimental observables
    theta: float, confidence parameter,

    Returns
    -------
    """
    #print("GRAD_BIOEN_LOG_POSTERIOR_BASE - NEEDS TO BE FIXED!!!!!")
    #print ("    forces shape", forces.shape)
    #print ("    w0 shape    ", w0.shape)
    #print ("    yTilde shape", yTilde.shape)
    #print ("    YTilde shape", YTilde.shape)
    #print ("    forces type ", str(type(forces)))
    #print ("    w0 type     ", str(type(w0)))
    #print ("    yTilde type ", str(type(yTilde)))
    #print ("    YTilde type ", str(type(YTilde)))
    forces = forces.T

    w = get_weights_from_forces(w0, yTilde, forces)

    yTildeAve = common.getAve(w, yTilde)


    B = np.dot(yTilde.T, (yTildeAve.T - YTilde).T)
    ratio = w / w0
    ind = np.where(np.logical_not(w > 0))[0]
    ratio[ind] = 1.
    A = (np.log(ratio) + 1.)
    C = (A * theta + B)
    E = np.asarray(C) * np.asarray(w)
    D = yTilde - yTildeAve[:, np.newaxis]
    F = np.dot(D, E)
    return np.asarray(F.T)[0]


# FORCES OPTIMIZER
def find_optimum(forcesInit, w0, y, yTilde, YTilde, theta, cfg):
    """
    For large number of structures N and M < N observables per structure, it is
    very efficient to optimize generalized forces.

    Parameters
    ----------
    forcesInit: 1xM matrix
    w0: array of length N
    y: MxN matrix, M observables calculate for the M structures
    yTilde: MxN matrix, M observables y_i / sigma_i for the M structures
    YTilde: 1xM matrix, experimental observables
    theta: float, confidence parameter,

    Returns
    -------
    wopt: array of length N, optimal weights for each structure in the ensemble
    yopt: 1xM matrix, observables calculated from optimal ensemble
    forces_opt: 1xM matrix, optimal forces

    """

    check_params_forces(forcesInit, w0, y, yTilde, YTilde)

    minimizer = cfg["minimizer"]
    caching = cfg["cache_ytilde_transposed"]
    if (caching == "auto"):
        m = yTilde.shape[0]
        n = yTilde.shape[1]
        caching = common.set_caching_heuristics(m, n)
    cfg["cache_ytilde_transposed"] = caching

    fmin_initial = c_bioen.bioen_log_posterior_forces(forcesInit,  w0, yTilde, YTilde, theta)
    if cfg["verbose"]:
        print("fmin_initial", fmin_initial)

    forces = forcesInit.copy()
    forces = forces.T

    start = time.time()

    if cfg["minimizer"].upper() == 'LIBLBFGS' or cfg["minimizer"].upper() == "LBFGS":
        common.print_highlighted("FORCES -- Library L-BFGS/C", cfg["verbose"])

        res = c_bioen.bioen_opt_lbfgs_forces(forces, w0, yTilde, YTilde, theta, cfg)

    elif cfg["minimizer"].upper() == 'GSL':
        common.print_highlighted("FORCES -- Library GSL/C", cfg["verbose"])

        res = c_bioen.bioen_opt_bfgs_forces(forces, w0, yTilde, YTilde, theta, cfg)

    elif cfg["minimizer"].upper() == 'SCIPY' and cfg["use_c_functions"] == True:
        common.print_highlighted("FORCES -- Library scipy/C", cfg["verbose"])

        caching = cfg["cache_ytilde_transposed"]

        if cfg["algorithm"].lower() == 'lbfgs' or cfg["algorithm"].lower() == "fmin_l_bfgs_b":
            common.print_highlighted('algorithm L-BFGS', cfg["verbose"])
            if cfg["verbose"]:
                print("\t", "=" * 25)
                print("\t", "caching_yTilde_transposed :     ", caching)
                print("\t", "epsilon                   :     ", cfg["params"]["epsilon"])
                print("\t", "pgtol                     :     ", cfg["params"]["pgtol"])
                print("\t", "maxiter                   :     ", cfg["params"]["max_iterations"])
                print("\t", "=" * 25)
            res = sopt.fmin_l_bfgs_b(c_bioen.bioen_log_posterior_forces,
                                     forces,
                                     args=(w0, yTilde, YTilde, theta, caching),
                                     fprime=c_bioen.grad_bioen_log_posterior_forces,
                                     epsilon=cfg["params"]["epsilon"],
                                     pgtol=cfg["params"]["pgtol"],
                                     maxiter=cfg["params"]["max_iterations"],
                                     disp=cfg["verbose"])

        elif cfg["algorithm"].lower() == 'bfgs' or cfg["algorithm"].lower() == "fmin_bfgs":
            common.print_highlighted('algorithm BFGS', cfg["verbose"])
            if cfg["verbose"]:
                print("\t", "=" * 25)
                print("\t", "caching_yTilde_transposed :     ", caching)
                print("\t", "epsilon                   :     ", cfg["params"]["epsilon"])
                print("\t", "gtol                      :     ", cfg["params"]["gtol"])
                print("\t", "maxiter                   :     ", cfg["params"]["max_iterations"])
                print("\t", "=" * 25)
            res = sopt.fmin_bfgs(c_bioen.bioen_log_posterior_forces,
                                 forces,
                                 args=(w0, yTilde, YTilde, theta, caching),
                                 fprime=c_bioen.grad_bioen_log_posterior_forces,
                                 epsilon=cfg["params"]["epsilon"],
                                 gtol=cfg["params"]["gtol"],
                                 maxiter=cfg["params"]["max_iterations"],
                                 disp=cfg["verbose"],
                                 full_output=True)

        elif cfg["algorithm"].lower() == 'cg' or cfg["algorithm"].lower() == 'fmin_cg':
            common.print_highlighted('algorithm CG', cfg["verbose"])
            if cfg["verbose"]:
                print("\t", "=" * 25)
                print("\t", "caching_yTilde_transposed :     ", caching)
                print("\t", "epsilon                   :     ", cfg["params"]["epsilon"])
                print("\t", "gtol                      :     ", cfg["params"]["gtol"])
                print("\t", "maxiter                   :     ", cfg["params"]["max_iterations"])
                print("\t", "=" * 25)
            res = sopt.fmin_cg(c_bioen.bioen_log_posterior_forces,
                               forces,
                               args=(w0, yTilde, YTilde, theta, caching),
                               fprime=c_bioen.grad_bioen_log_posterior_forces,
                               epsilon=cfg["params"]["epsilon"],
                               gtol=cfg["params"]["gtol"],
                               maxiter=cfg["params"]["max_iterations"],
                               disp=cfg["verbose"],
                               full_output=True)

        else:
            raise RuntimeError("Method '" + cfg["algorithm"] + "' not recognized for scipy/c library (valid values =  'lbfgs', 'bfgs', 'cg' ) ")

    elif cfg["minimizer"].upper() == 'SCIPY' and cfg["use_c_functions"] == False:
        common.print_highlighted("FORCES -- Library scipy/PY", cfg["verbose"])

        if cfg["algorithm"].lower() == 'lbfgs' or cfg["algorithm"].lower() == "fmin_l_bfgs_b":
            common.print_highlighted('algorithm L-BFGS', cfg["verbose"])
            if cfg["verbose"]:
                print("\t", "=" * 25)
                print("\t", "epsilon     ", cfg["params"]["epsilon"])
                print("\t", "pgtol       ", cfg["params"]["pgtol"])
                print("\t", "maxiter     ", cfg["params"]["max_iterations"])
                print("\t", "=" * 25)


            #### HERE CRASHES with ndarray  ###
            #print("#########################")
            #print("type forces", str(type(forces)))
            #print("type w0    ", str(type(w0)))
            #print("type yTilde", str(type(yTilde)))
            #print("type YTilde", str(type(YTilde)))
            #print("type theta ", str(type(theta)))
            #print("#########################")
            #print("shape forces", forces.shape)
            #print("shape w0    ", w0.shape)
            #print("shape yTilde", yTilde.shape)
            #print("shape YTilde", YTilde.shape)
            #print("#########################")
            res = sopt.fmin_l_bfgs_b(bioen_log_posterior_base,
                                     forces,
                                     args=(w0, yTilde, YTilde, theta),
                                     fprime=grad_bioen_log_posterior_base,
                                     epsilon=cfg["params"]["epsilon"],
                                     pgtol=cfg["params"]["pgtol"],
                                     maxiter=cfg["params"]["max_iterations"],
                                     disp=cfg["verbose"])

        elif cfg["algorithm"].lower() == 'bfgs' or cfg["algorithm"].lower() == "fmin_bfgs":
            common.print_highlighted('algorithm BFGS', cfg["verbose"])
            if cfg["verbose"]:
                print("\t", "=" * 25)
                print("\t", "epsilon     ", cfg["params"]["epsilon"])
                print("\t", "gtol        ", cfg["params"]["gtol"])
                print("\t", "maxiter     ", cfg["params"]["max_iterations"])
                print("\t", "=" * 25)
            res = sopt.fmin_bfgs(bioen_log_posterior_base,
                                 forces,
                                 args=(w0, yTilde, YTilde, theta),
                                 fprime=grad_bioen_log_posterior_base,
                                 epsilon=cfg["params"]["epsilon"],
                                 gtol=cfg["params"]["gtol"],
                                 maxiter=cfg["params"]["max_iterations"],
                                 disp=cfg["verbose"],
                                 full_output=True)

        elif cfg["algorithm"].lower() == 'cg' or cfg["algorithm"].lower() == 'fmin_cg':
            common.print_highlighted('algorithm CG', cfg["verbose"])
            if cfg["verbose"]:
                print("\t", "=" * 25)
                print("\t", "epsilon     ", cfg["params"]["epsilon"])
                print("\t", "gtol        ", cfg["params"]["gtol"])
                print("\t", "maxiter     ", cfg["params"]["max_iterations"])
                print("\t", "=" * 25)
            res = sopt.fmin_cg(bioen_log_posterior_base,
                               forces,
                               args=(w0, yTilde, YTilde, theta),
                               fprime=grad_bioen_log_posterior_base,
                               epsilon=cfg["params"]["epsilon"],
                               gtol=cfg["params"]["gtol"],
                               maxiter=cfg["params"]["max_iterations"],
                               disp=cfg["verbose"],
                               full_output=True)
        else:
            raise RuntimeError("Method '" + cfg["algorithm"] + "' not recognized for scipy/py library (valid values =  'lbfgs', 'bfgs', 'cg' ) ")

    else:
        raise RuntimeError("Library " + cfg["minimizer"] + " not recognized (valid values =  'LIBLBFGS', 'GSL', 'scipy', 'scipy' ) ")

    end = time.time()

    if cfg["verbose"]:
        print('time elapsed ', (end - start))

    forces_opt = res[0]
    fmin_final = res[1]

    forces_opt = forces_opt.T
    wopt = get_weights_from_forces(w0, yTilde, forces_opt)

    yopt = common.getAve(wopt, y)

    if cfg["verbose"]:
        print("========================")
        print("fmin_initial           =", fmin_initial)
        print("fmin_final             =", fmin_final)
        print("========================")

    S, chiSqr = bioen_chi2_s_forces(forces_opt, w0, yTilde, YTilde)

    return wopt, yopt, forces_opt, fmin_initial, fmin_final, chiSqr, S
