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
    elif any(extension in w_setting[-4:] for extension in [".dat", ".txt"]):
        # complains if file does not exist
        try:
            wtmp = np.loadtxt(w_setting, unpack=True)
        except Exception as e:
            print(e)
            print('ERROR: Please provide the correct path to your weights file.' +
                  'Right now it is: \'{}\'.'.format(w_setting))
            raise
        wtmp[wtmp == 0.0] = 1e-150
        wsum = np.sum(wtmp)

        # complains if input by the user is inconsistent
        if len(wtmp) != nmodels:
            msg = 'ERROR: Please provide the same number of ensemble members ' +\
                  '(--number_of_models {}) as number of weights in your '.format(nmodels) +\
                  'file \'{}\' ({}).'.format(w_setting, len(wtmp))
            raise RuntimeError(msg)

        return (np.matrix(np.array(wtmp)/np.array(wsum))).T
    # complains if not the correct input is provided
    else:
        msg = 'ERROR: Please provide information on weights (e.g. \'uniform\', \'random\' or ' +\
              'define a file in .dat or .txt format).'
        raise RuntimeError(msg)


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
    
    Warnings
    --------
    The current implementation returns the negative Kullback-Leibler divergence.
    For ellbow plots, you have to take minus the output of the function. This likely
    to be changed in the future so that the function returns the Kullback-Leibler divergence. 
    
    References
    ----------
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence      
    """
    return - np.sum(weights.T * np.log(weights / w0))


def is_float(text):
    """
    Checks if string can be converted to float.

    Parameters
    ----------
    text: string

    Returns
    -------
    float/boolean: float if text can be converted to float, False if not
    """
    try:
        return float(text)
    except ValueError:
        return False


def load_lines(fn):
    """
    Load content of a file and ignore '#'.
    Here: file lines contain also IDs or label information, which can
    be in string format.

    Parameters
    ----------
    fn: string,
        filename

    Returns
    -------
    lines: array,
        array of strings
    """
    lines_all = open(fn, 'r').readlines()
    lines = []
    for line in lines_all:
        if not line.startswith('#'):
            lines.append(line.split())
    return lines
