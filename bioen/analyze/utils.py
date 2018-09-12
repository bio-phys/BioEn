from __future__ import print_function
import numpy as np
import scipy.optimize as sopt
import sys


def get_weights(w_setting, nmodels):
    """
    Provides weights (either reference or initial weights).

    Parameters
    ----------
    w_setting: string, either uniform, random or file
    nmodels: int, number of models/conformers

    Returns
    -------
    weights: array-like, weights (len(weights)=nmodels)
    """
    if w_setting == 'uniform':
        return get_uniform_weights(nmodels)
    elif w_setting == 'random':
        return get_rdm_weights(nmodels)
    elif ".dat" in w_setting:
        wtmp = np.loadtxt(w_setting, unpack=True)
        wtmp[wtmp == 0.0] = 1e-150
        wsum = np.sum(wtmp)
        return (np.matrix(np.array(wtmp)/np.array(wsum))).T
    else:
        raise RuntimeError('ERROR: Please provide information on weights '
                           '(e.g. \'uniform\', \'random\' or define a file\').')


def get_uniform_weights(nmodels):
    """
    Provides uniformly distributed weights.

    Parameters
    ----------
    nmodels: int, number of models/conformers

    Returns
    -------
    weights: array-like, weights (len(weights)=nmodels)
    """
    return (np.matrix(np.ones(nmodels))/nmodels).T


def get_rdm_weights(nmodels):
    """
    Provides randomly distributed weights.

    Parameters
    ----------
    nmodels: int, number of models/conformers

    Returns
    -------
    weights: array-like, weights (len(weights)=nmodels)
    """
    wtmp = np.random.uniform(0, 0.001, size=nmodels)
    wsum = np.sum(wtmp)
    return (np.matrix(np.array(wtmp)/np.array(wsum))).T


def get_entropy(w0, weights):
    """
    Provides relative entropy (S).

    Parameters
    ----------
    w0: array-like, reference weights
    weights: array-like, current weights

    Returns
    -------
    S: float, relative entropy
    """
    return - np.sum(weights.T * np.log(weights / w0))


def is_float(text):
    """
    Checks if string can be converted to float.

    Parameters
    ----------
    text: string

    Returns:
    --------
    float/boolean: float if text can be converted to float, False if not
    """
    try:
        return float(text)
    except ValueError:
        return False


def load_lines(fn):
    lines_all = open(fn, 'r').readlines()
    lines = []
    for line in lines_all:
        if not line.startswith('#'):
            lines.append(line)
    return lines
