from __future__ import print_function
import numpy as np


def chiSqrTerm(w, yTilde, YTilde):
    """
    Legacy Python version of chiSqrTerm.

    Parameters
    ----------
    w: array like, N  elements
    yTilde: array_like, MxN matrix
    YTilde: array_like, vector with M components

    Returns
    -------

    """
    result = 0.

    #print ("Shape yTilde", yTilde.shape)
    #print ("Shape yTilde", str(type(yTilde)))
    #print ("Shape YTilde", YTilde.shape)
    #print ("Shape YTilde", str(type(YTilde)))
    #print ("Shape w     ", w.shape)
    #print ("Shape w     ", str(type(w)))
    if (isinstance(yTilde, np.matrixlib.defmatrix.matrix)):
        v = (yTilde * w).T - YTilde
        result = .5 * (v * v.T)[0, 0]
    else:
        v = np.dot(yTilde,w).T - YTilde
        #print ("Shape v     ", v.shape)
        #print ("Shape v     ", str(type(v)))
        result = np.dot(.5,np.dot(v,v.T))[0, 0]
    
    #v = (yTilde * w).T - YTilde
    #result = .5 * (v * v.T)[0, 0]

    return result



def getAve(w, y):
    """
    Legacy Python version of getAve.

    Parameters
    ----------
    w: array like, N  elements (obtained from getWeights(g))
    y: array like,

    Returns
    -------
    array like
    """
    #result = []
    if (isinstance(y, np.matrixlib.defmatrix.matrix)):
        result = np.asarray((y * w).T)[0]
    else:
        result = np.dot(y, w)[:,0]
  
    #result = np.asarray((y * w).T)[0]
    return result


def print_highlighted(str, verbose=True):
    """
    Decorate a string to improve readability

    Parameters
    ----------
    str:
    verbose: True as default. Prints highlighted str

    """
    if verbose:
        n = len(str)
        print("-"*n)
        print(str)
        print("-"*n)


def set_caching_heuristics(m, n):
    """
    Function that calculates size required for M:observables and N:structures
    (size = sizeof(double) * m * n ) and evalutes this size for a fixed
    threshold (2^30 * 8).
    This is used to decide wheter caching functionality has to be enabled or
    not when 'cache_ytilde_transposed' is configured in 'auto' mode.

    Parameters
    ----------
    m: size of observables
    n: size of structures

    Returns
    -------
    bool: True (acceptable memory for caching)

    """
    max_acceptable_mem = 8 * 2**30
    sizeof_double = 8
    if (m * n * sizeof_double > max_acceptable_mem):
        return False

    return True
