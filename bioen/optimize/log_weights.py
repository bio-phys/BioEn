"""Python interface for the log_weights method
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


# LOGWEIGHTS general functions
def bioen_log_prior(w, s, g, G, theta):
    """
    Legacy Python version of bioen_log_prior

    Parameters
    ---------
    w: array like, N  elements (obtained from getWeights(g))
    s: double, (obtained from getWeights(g))
    g: array_like, N log weights (initial log-weights)
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    theta: float, confidene parameter

    Returns
    -------
    double:
      log_prior value
    """
    w0, s0 = getWeights(G)
    return (theta * ((g.T * w) - (G.T * w) - np.log(s) + np.log(s0)))[0, 0]


def init_log_weights(w0):
    """
    Observed values drawn from normal distribution.

    Parameters
    ---------
    w0:

    Returns
    -------
    gPrime:
    g:
    G:
    GInit:
    """
    G = getGs(w0)
    GInit = getGs(np.matrix(w0))
    g = GInit.copy()
    gPrime = np.asarray(g[:-1].T)[0]
    return gPrime, g, G, GInit


def getWeights(g):
    """
    Legacy Python version of getWeights.

    Parameters
    ---------
    g: array_like, N log weights (initial log-weights)

    Returns
    -------
    w: array like, N elems.
    s: double
    """
    tmp = np.exp(g)
    s = tmp.sum()
    w = (np.matrix(tmp / s))
    return w, s


def getGs(w):
    """
    Legacy Python version of getGs.

    Parameters
    ---------
    w: array like, N  elements (obtained from getWeights(g))

    Returns
    -------
    array like, N elems.
    """
    g = np.log(w)
    g -= g[-1, 0]
    return g


def getWOpt(G, gPrimeOpt):
    """
    Legacy Python version of getWOpt.
    Parameters
    ---------
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    gPrimeOpt:

    Returns
    -------
    wopt: array like, N elems.
    """
    gopt = G.copy()
    gopt[:] = 0.
    gopt[:, 0] = gPrimeOpt[:, np.newaxis]
    wopt, s = getWeights(gopt)
    return wopt


def grad_chiSqrTerm(gPrime, g, G, yTilde, YTilde, theta):
    """
    Legacy Python version of grad_chiSqrTerm.

    Parameters
    ---------
    gPrime: array_like, current log weights
    g: array_like, N log weights (initial log-weights)
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    yTilde: array_like, MxN matrix
    YTilde: array_like, vector with M components
    theta: float, confidene parameter

    Returns
    -------
    """
    g[:-1, 0] = gPrime[:, np.newaxis]
    g[-1, 0] = 0
    w, s = getWeights(g)
    tmp = np.zeros(w.shape[0])
    yTildeAve = yTilde * w
    for mu in range(w.shape[0]):
        tmp[mu] = w[mu] * (yTildeAve.T - YTilde) * (yTilde[:, mu] - yTildeAve)
    tmp = np.matrix(tmp)
    return np.asarray(tmp)[0][:-1]


def check_params_logweights(GInit, G, y, yTilde, YTilde):
    """
    Check the proper dimensionality of the logweights parameters.
    The reference dimensions are based on the MxN matrix yTilde

    Parameters
    ---------
    gInit: array_like, N log weights (initial log-weights)
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    y: array_like, MxN matrix
    yTilde: array_like, MxN matrix
    YTilde: array_like, vector with M components

    Notes
    -----
    Execution is aborted if dimensionality is not well defined.
    """
    numdim = yTilde.ndim
    m = yTilde.shape[0]
    n = yTilde.shape[1]
    error = 0

    if (GInit.ndim != numdim or GInit.shape[0] != n or GInit.shape[1] != 1):
        print("Correct form: GInit (", m, ",1)")
        print("Current form: GInit (", GInit.shape[0], ",", GInit.shape[1], ")")
        error = 1

    if (G.ndim != numdim or G.shape[0] != n or G.shape[1] != 1):
        print("Correct form: G (", n, ",1)")
        print("Current form: G (", G.shape[0], ",", G.shape[1], ")")
        error = 1

    if (y.ndim != numdim or y.shape[0] != m or y.shape[1] != n):
        print("Correct form: y (", m, ",", n, ")")
        print("Current form: y (", y.shape[0], ",", y.shape[1], ")")
        error = 1

    if (YTilde.ndim != numdim or YTilde.shape[0] != 1 or YTilde.shape[1] != m):
        print("Correct form: YTilde (1,", m, ")")
        print("Current form: YTilde (", YTilde.shape[0], ",", YTilde.shape[1], ")")
        error = 1

    if (error == 1):
        raise RuntimeError("EXIT: Error/s on arguments dimensionality for the 'log_weigths' method")

    return


# LOGWEIGHTS function interface selector
def bioen_log_posterior(gPrime, g, G, yTilde, YTilde, theta, use_c=True, caching=False):
    """
    Interface for the objective function log_weights. Selector of C/GSL or Python

    Parameters
    ---------
    gPrime: array_like, current log weights
    g: array_like, N log weights (initial log-weights)
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    yTilde: array_like, MxN matrix
    YTilde: array_like, vector with M components
    theta: float, confidene parameter

    Returns
    -------
    double:
      Objective function value (fmin)
    """
    if use_c:
        val = c_bioen.bioen_log_posterior_logw(gPrime, g, G, yTilde, YTilde, theta, caching=caching)
    else:
        val = bioen_log_posterior_base(gPrime, g, G, yTilde, YTilde, theta)
    return val


# LOGWEIGHTS gradient interface selector
def grad_bioen_log_posterior(gPrime, g, G, yTilde, YTilde, theta, use_c=True, caching=False):
    """
    Interface for the log_weights gradient. Selector of C/GSL or Python

    Parameters
    ----------
    gPrime: array_like, current log weights
    g: array_like, N log weights (initial log-weights)
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    yTilde: array_like, MxN matrix
    YTilde: array_like, vector with M components
    theta: float, confidene parameter

    Returns
    -------
    array_like, vector with N components
    """
    if use_c:
        grad = c_bioen.grad_bioen_log_posterior_logw(gPrime, g, G, yTilde, YTilde, theta, caching=caching)
    else:
        grad = grad_bioen_log_posterior_base(gPrime, g, G, yTilde, YTilde, theta)

    print("grad", grad)
    return grad


# LOGWEIGHTS function
def bioen_log_posterior_base(gPrime, g, G, yTilde, YTilde, theta):
    """
    Legacy Python version of bioen_log_posterior

    Parameters
    -----------
    gPrime: array_like, current log weights
    g: array_like, N log weights (initial log-weights)
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    yTilde: array_like, MxN matrix
    YTilde: array_like, vector with M components
    theta: float, confidene parameter

    Returns
    --------
    L: BioEn loglikelihood
    """
    g[:, 0] = gPrime[:, np.newaxis]
    w, s = getWeights(g)
    val1 = bioen_log_prior(w, s, g, G, theta)
    val2 = common.chiSqrTerm(w, yTilde, YTilde)
    return val1 + val2


# LOGWEIGHTS gradient
def grad_bioen_log_posterior_base(gPrime, g, G, yTilde, YTilde, theta):
    """
    Legacy Python version of grad_bioen_log_posterior

    Parameters
    ---------
    gPrime: array_like, current log weights
    g: array_like, N log weights (initial log-weights)
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    yTilde: array_like, MxN matrix
    YTilde: array_like, vector with M components
    theta: float, confidene parameter

    Return
    ------
    array_like, vector with N components
    """

    g[:, 0] = gPrime[:, np.newaxis]

    w, s = getWeights(g)

    tmp = np.zeros(w.shape[0])
    yTildeAve = yTilde * w
    for mu in range(w.shape[0]):
        tmp[mu] = w[mu] * (yTildeAve.T - YTilde) * (yTilde[:, mu] - yTildeAve)
    tmp = np.matrix(tmp)
    value = np.asarray((np.asarray(w.T) * np.asarray(theta * (g - g.T * w - G + G.T * w).T) + tmp))[0][:]
    return value


def find_optimum(GInit, G, y, yTilde, YTilde, theta, cfg):
    """
    Find the optimal solution for the BioEn problem using numerical optimization.

    Gradients are calculated analytically, which typically gives a large speed up.

    Parameters
    ----------
    GInit: array_like, vector with N components, starting value for optimization
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    y: array_like, MxN matrix
    yTilde: array_like, MxN matrix
    YTilde: array_like, vector with M components
    theta: float, confidence parameter

    Returns
    -------
    wopt: optimized weights
    yopt: measurement value after refinement
    fmin_initial: float, starting negative log-likelihood (optional)
    fmin_final: float, final negative log-lilelihood (optional)
    """

    check_params_logweights(GInit, G, y, yTilde, YTilde)

    minimizer = cfg["minimizer"]
    caching = cfg["cache_ytilde_transposed"]
    if (caching == "auto"):
        m = yTilde.shape[0]
        n = yTilde.shape[1]
        caching = common.set_caching_heuristics(m, n)
    cfg["cache_ytilde_transposed"] = caching

    g = GInit.copy()
    gPrime = np.asarray(g[:].T)[0]

    fmin_initial = bioen_log_posterior_base(gPrime, g, G, yTilde, YTilde, theta)
    if cfg["verbose"]:
        print("fmin_initial", fmin_initial)

    start = time.time()

    if cfg["minimizer"].upper() == 'LIBLBFGS' or cfg["minimizer"].upper() == "LBFGS":
        common.print_highlighted("LOGW -- Library L-BFGS/C", cfg["verbose"])

        res = c_bioen.bioen_opt_lbfgs_logw(gPrime, G, yTilde, YTilde, theta, cfg)

    elif cfg["minimizer"].upper() == 'GSL':
        common.print_highlighted("LOGW -- Library GSL/C", cfg["verbose"])

        res = c_bioen.bioen_opt_bfgs_logw(gPrime, G, yTilde, YTilde, theta, cfg)

    elif cfg["minimizer"].upper() == 'SCIPY' and cfg["use_c_functions"] == True:
        common.print_highlighted("LOGW -- Library scipy/C", cfg["verbose"])

        caching = cfg["cache_ytilde_transposed"]

        if cfg["algorithm"].lower() == 'lbfgs' or cfg["algorithm"].lower() == "fmin_l_bfgs_b":

            common.print_highlighted('method L-BFGS', cfg["verbose"])
            if cfg["verbose"]:
                print("\t", "=" * 25)
                print("\t", "caching_yTilde_transposed :     ", caching)
                print("\t", "epsilon                   :     ", cfg["params"]["epsilon"])
                print("\t", "pgtol                     :     ", cfg["params"]["pgtol"])
                print("\t", "maxiter                   :     ", cfg["params"]["max_iterations"])
                print("\t", "=" * 25)

            res = sopt.fmin_l_bfgs_b(c_bioen.bioen_log_posterior_logw,
                                     gPrime,
                                     args=(g, G, yTilde, YTilde, theta, caching),
                                     fprime=c_bioen.grad_bioen_log_posterior_logw,
                                     epsilon=cfg["params"]["epsilon"],
                                     pgtol=cfg["params"]["pgtol"],
                                     maxiter=cfg["params"]["max_iterations"],
                                     disp=cfg["verbose"])

        elif cfg["algorithm"].lower() == 'bfgs' or cfg["algorithm"].lower() == "fmin_bfgs":

            common.print_highlighted('method BFGS', cfg["verbose"])
            if cfg["verbose"]:
                print("\t", "=" * 25)
                print("\t", "caching_yTilde_transposed :     ", caching)
                print("\t", "epsilon                   :     ", cfg["params"]["epsilon"])
                print("\t", "gtol                      :     ", cfg["params"]["gtol"])
                print("\t", "maxiter                   :     ", cfg["params"]["max_iterations"])
                print("\t", "=" * 25)

            res = sopt.fmin_bfgs(c_bioen.bioen_log_posterior_logw,
                                 gPrime,
                                 args=(g, G, yTilde, YTilde, theta, caching),
                                 fprime=c_bioen.grad_bioen_log_posterior_logw,
                                 epsilon=cfg["params"]["epsilon"],
                                 gtol=cfg["params"]["gtol"],
                                 maxiter=cfg["params"]["max_iterations"],
                                 disp=cfg["verbose"],
                                 full_output=True)

        elif cfg["algorithm"].lower() == 'cg' or cfg["algorithm"].lower() == 'fmin_cg':

            common.print_highlighted('method CG', cfg["verbose"])
            if cfg["verbose"]:
                print("\t", "=" * 25)
                print("\t", "caching_yTilde_transposed :     ", caching)
                print("\t", "epsilon                   :     ", cfg["params"]["epsilon"])
                print("\t", "gtol                      :     ", cfg["params"]["gtol"])
                print("\t", "maxiter                   :     ", cfg["params"]["max_iterations"])
                print("\t", "=" * 25)

            res = sopt.fmin_cg(c_bioen.bioen_log_posterior_logw,
                               gPrime,
                               args=(g, G, yTilde, YTilde, theta, caching),
                               fprime=c_bioen.grad_bioen_log_posterior_logw,
                               epsilon=cfg["params"]["epsilon"],
                               gtol=cfg["params"]["gtol"],
                               maxiter=cfg["params"]["max_iterations"],
                               disp=cfg["verbose"],
                               full_output=True)

        else:
            raise RuntimeError("Method '" + cfg["algorithm"] + "' not recognized for scipy/c library (valid values =  'lbfgs', 'bfgs', 'cg' ) ")

    elif cfg["minimizer"].upper() == 'SCIPY' and cfg["use_c_functions"] == False:

        common.print_highlighted("LOGW -- Library scipy/PY", cfg["verbose"])

        if cfg["algorithm"].lower() == 'lbfgs' or cfg["algorithm"].lower() == "fmin_l_bfgs_b":

            common.print_highlighted('method L-BFGS', cfg["verbose"])
            if cfg["verbose"]:
                print("\t", "=" * 25)
                print("\t", "epsilon     ", cfg["params"]["epsilon"])
                print("\t", "pgtol       ", cfg["params"]["pgtol"])
                print("\t", "maxiter     ", cfg["params"]["max_iterations"])
                print("\t", "=" * 25)

            res = sopt.fmin_l_bfgs_b(bioen_log_posterior_base,
                                     gPrime,
                                     args=(g, G, yTilde, YTilde, theta),
                                     fprime=grad_bioen_log_posterior_base,
                                     epsilon=cfg["params"]["epsilon"],
                                     pgtol=cfg["params"]["pgtol"],
                                     maxiter=cfg["params"]["max_iterations"],
                                     disp=cfg["verbose"])

        elif cfg["algorithm"].lower() == 'bfgs' or cfg["algorithm"].lower() == "fmin_bfgs":

            common.print_highlighted('method BFGS', cfg["verbose"])
            if cfg["verbose"]:
                print("\t", "=" * 25)
                print("\t", "epsilon     ", cfg["params"]["epsilon"])
                print("\t", "gtol        ", cfg["params"]["gtol"])
                print("\t", "maxiter     ", cfg["params"]["max_iterations"])
                print("\t", "=" * 25)

            res = sopt.fmin_bfgs(bioen_log_posterior_base,
                                 gPrime,
                                 args=(g, G, yTilde, YTilde, theta),
                                 fprime=grad_bioen_log_posterior_base,
                                 epsilon=cfg["params"]["epsilon"],
                                 gtol=cfg["params"]["gtol"],
                                 maxiter=cfg["params"]["max_iterations"],
                                 disp=cfg["verbose"],
                                 full_output=True)

        elif cfg["algorithm"].lower() == 'cg' or cfg["algorithm"].lower() == 'fmin_cg':

            common.print_highlighted('method CG', cfg["verbose"])
            if cfg["verbose"]:
                print("\t", "=" * 25)
                print("\t", "epsilon     ", cfg["params"]["epsilon"])
                print("\t", "gtol        ", cfg["params"]["gtol"])
                print("\t", "maxiter     ", cfg["params"]["max_iterations"])
                print("\t", "=" * 25)

            res = sopt.fmin_cg(bioen_log_posterior_base,
                               gPrime,
                               args=(g, G, yTilde, YTilde, theta),
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

    gopt = res[0]
    fmin_final = res[1]

    wopt = getWOpt(G, gopt)
    yopt = common.getAve(wopt, y)

    if cfg["verbose"]:
        print("========================")
        print("fmin_initial  = ", fmin_initial)
        print("fmin_final    = ", fmin_final)
        print("========================")

    return wopt, yopt, gopt, fmin_initial, fmin_final
