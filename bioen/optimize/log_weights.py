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
    ----------
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

    if (isinstance(G, np.matrixlib.defmatrix.matrix)):
        #result = (theta * ((g.T * w) - (G.T * w) - np.log(s) + np.log(s0)))[0, 0]
        var1= g.T * w
        var2= G.T * w
        var3= np.log(s)
        var4= np.log(s0)

        oper1 = var1 - var2
        oper2 = oper1 - var3
        oper3 = oper2 + var4

        mul1 = theta * oper3

        result = mul1[0,0]

    else:

        var1= np.dot(g.T,w)
        var2= np.dot(G.T,w)
        var3= np.log(s)
        var4= np.log(s0)

        oper1 = var1 - var2
        oper2 = oper1 - var3
        oper3 = oper2 + var4

        mul1 = theta * oper3

        result = mul1[0,0]


    return result


def init_log_weights(w0):
    """
    Observed values drawn from normal distribution.

    Parameters
    ----------
    w0:

    Returns
    -------
    gPrime:
    g:
    G:
    GInit:
    """
    G = getGs(w0)
    GInit = getGs(np.array(w0))
    g = GInit.copy()
    gPrime = np.asarray(g[:-1].T)[0]
    return gPrime, g, G, GInit


def getWeights(g):
    """
    Legacy Python version of getWeights.

    Parameters
    ----------
    g: array_like, N log weights (initial log-weights)

    Returns
    -------
    w: array like, N elems.
    s: double
    """
    tmp = np.exp(g)
    s = tmp.sum()
    w = (np.array(tmp / s))

    return w, s


def getGs(w):
    """
    Legacy Python version of getGs.

    Parameters
    ----------
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
    ----------
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    gPrimeOpt:

    Returns
    -------
    wopt: array like, N elems.
    """

    if (isinstance(G, np.matrixlib.defmatrix.matrix)):
        gopt = G.copy()
        gopt[:] = 0.
        gopt[:, 0] = gPrimeOpt[:, np.newaxis]
        wopt, s = getWeights(gopt)
    else:
        gopt = G.copy()
        gopt[:] = 0.
        gopt = np.expand_dims(gPrimeOpt,axis=1)
        wopt, s = getWeights(gopt)

    #gopt = G.copy()
    #gopt[:] = 0.
    #gopt[:, 0] = gPrimeOpt[:, np.newaxis]
    #wopt, s = getWeights(gopt)


    return wopt



def grad_chiSqrTerm(gPrime, g, G, yTilde, YTilde, theta):
    """
    Legacy Python version of grad_chiSqrTerm.

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
    """
    g[:-1, 0] = gPrime[:, np.newaxis]
    g[-1, 0] = 0
    w, s = getWeights(g)
    tmp = np.zeros(w.shape[0])
    yTildeAve = yTilde * w
    for mu in range(w.shape[0]):
        tmp[mu] = w[mu] * (yTildeAve.T - YTilde) * (yTilde[:, mu] - yTildeAve)
    tmp = np.array(tmp)
    return np.asarray(tmp)[0][:-1]


def check_params_logweights(GInit, G, y, yTilde, YTilde):
    """
    Check the proper dimensionality of the logweights parameters.
    The reference dimensions are based on the (m, n) matrix yTilde

    Parameters
    ----------
    GInit : (n, 1)
    G : (n, 1)
    y : (m, n)
    YTilde : (1, m)

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
    if GInit.shape != (n, 1):
        _pretty_print("GInit", (n, 1), GInit.shape)
        error = True

    if G.shape != (n, 1):
        _pretty_print("G", (n, 1), G.shape)
        error = True

    if y.shape != (m, n):
        _pretty_print("y", (m, n), y.shape)
        error = True

    if YTilde.shape != (1, m):
        _pretty_print("YTilde", (1, m), YTilde.shape)
        error = True

    if error:
        raise ValueError("arguments dimensionality for the 'log_weights' method are wrong")


# LOGWEIGHTS function interface selector
def bioen_log_posterior(gPrime, g, G, yTilde, YTilde, theta, use_c=True, caching=False):
    """
    Interface for the objective function log_weights. Selector of C/GSL or Python

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

    return grad


# LOGWEIGHTS function
def bioen_log_posterior_base(gPrime, g, G, yTilde, YTilde, theta):
    """
    Legacy Python version of bioen_log_posterior

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
    L: BioEn loglikelihood
    """

    if (isinstance(g, np.matrixlib.defmatrix.matrix)):
        g[:, 0] = gPrime[:, np.newaxis]
        w, s = getWeights(g)
        val1 = bioen_log_prior(w, s, g, G, theta)
        val2 = common.chiSqrTerm(w, yTilde, YTilde)
    else:
        gPrime = np.expand_dims(gPrime, axis=1)
        g[:, 0] = gPrime[:, 0]
        #g[:, 0] = gPrime[:, np.newaxis]
        w, s = getWeights(g)
        val1 = bioen_log_prior(w, s, g, G, theta)
        val2 = common.chiSqrTerm(w, yTilde, YTilde)

    #g[:, 0] = gPrime[:, np.newaxis]
    #w, s = getWeights(g)
    #val1 = bioen_log_prior(w, s, g, G, theta)
    #val2 = common.chiSqrTerm(w, yTilde, YTilde)

    result = val1 + val2

    return result
    #return val1 + val2


# LOGWEIGHTS gradient
def grad_bioen_log_posterior_base(gPrime, g, G, yTilde, YTilde, theta):
    """
    Legacy Python version of grad_bioen_log_posterior

    Parameters
    ----------
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

    value = 0.
    if (isinstance(yTilde, np.matrixlib.defmatrix.matrix)):
        g[:, 0] = gPrime[:, np.newaxis]
        w, s = getWeights(g)

        tmp = np.zeros(w.shape[0])
        yTildeAve = yTilde * w
        for mu in range(w.shape[0]):
            #tmp[mu] = w[mu] * (yTildeAve.T - YTilde) * (yTilde[:, mu] - yTildeAve)
            var1 = w[mu]
            var2 = (yTildeAve.T - YTilde)
            inter = yTilde[:,mu]
            var3 = (inter - yTildeAve)
            var4 = var1 * var2
            var5 = var4 * var3
            var5 = var5[0,0]
            tmp[mu] = var5

        tmp = np.array(tmp)

        value = np.asarray((np.asarray(w.T) * np.asarray(theta * ((g - (g.T * w)) - (G + (G.T * w))).T) + tmp))[0][:]
    else:
        g = gPrime[:, np.newaxis]
        w, s = getWeights(g)

        tmp = np.zeros(w.shape[0])
        yTildeAve = np.dot (yTilde,w)
        for mu in range(w.shape[0]):
            #tmp[mu] = w[mu] * (yTildeAve.T - YTilde) * (yTilde[:, mu] - yTildeAve)
            var1 = w[mu]
            var2 = (yTildeAve.T - YTilde)
            inter = yTilde[:,mu]
            inter = inter[:,np.newaxis]
            var3 = (inter - yTildeAve)
            var4 = np.dot(var1, var2)
            var5 = np.dot(var4, var3)
            tmp[mu] = var5

        op1 =  g - np.dot(g.T,w)
        op2 =  G + np.dot(G.T,w)
        op3 = ( w.T * np.dot(theta,op1 - op2).T) + tmp

        value = op3[0,:]


    #g[:, 0] = gPrime[:, np.newaxis]
    #w, s = getWeights(g)

    #tmp = np.zeros(w.shape[0])
    #yTildeAve = yTilde * w
    #for mu in range(w.shape[0]):
    #    tmp[mu] = w[mu] * (yTildeAve.T - YTilde) * (yTilde[:, mu] - yTildeAve)
    #tmp = np.array(tmp)
    #value = np.asarray((np.asarray(w.T) * np.asarray(theta * (g - g.T * w - G + G.T * w).T) + tmp))[0][:]


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
