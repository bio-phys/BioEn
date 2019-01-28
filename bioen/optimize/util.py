"""Python general utilities
"""

import os
import sys
import yaml
import numpy as np

from .ext import c_bioen


def library_gsl():
    """
    Check for the availability of the GSL library

    Parameters
    ----------

    Returns
    -------
    bool: True if GSL enabled


    """
    return c_bioen.library_gsl()


def library_lbfgs():
    """
    Check for the availability of the LBFGS library

    Parameters
    ----------

    Returns
    -------
    bool: True if LBFGS enabled

    """
    return c_bioen.library_lbfgs()


def compute_relative_difference_for_values(a, b):
    """
    Computes relative difference for values

    Parameters
    ----------
    a: value
    b: reference

    Returns
    -------
    d: relative difference

    """
    if (b == 0):
        return abs(a)
    d = abs(a - b)/abs(b)
    return d


def compute_relative_difference_for_arrays(a, b):
    """
    Computes relative difference for arrays

    Parameters
    ----------
    a: array value
    b: array reference

    Returns
    -------
    d: array with relative differences
    idx: index of maximum difference value

    """
    nonzero_idx = np.where(b != 0.0)
    nonzero_msk = np.zeros(b.shape, dtype=bool)
    nonzero_msk[nonzero_idx] = True
    d_array = np.absolute(a[nonzero_msk] - b[nonzero_msk])/np.absolute(b[nonzero_msk])
    d = np.max(d_array)
    idx = np.where(d_array == d)[0][0]
    return d, idx


def load_template_config_yaml(file_name, minimizer):
    """
    Loads the bioen configuration for a specified minimizer from a yaml file.

    Parameters
    ----------
    file_name: file containing the configuration
    minimizer: specific section to be loaded

    Returns
    -------
    params: a map structure containing the configuration

    """

    minimizer = minimizer.lower()

    with open(file_name, "r") as fp:
        cfg = yaml.load(fp)

    packed_params = {}

    # 3
    debug = cfg["general"]["debug"]
    # 4
    verbose = cfg["general"]["verbose"]

    # 6  #params per "minimizer"
    params = {}
    for modif in cfg[minimizer]:
        params[modif] = cfg[minimizer][modif]

    # 7
    n_threads = cfg["c_functions"]["n_threads"]
    # 8
    cache_ytilde_transposed = cfg["c_functions"]["cache_ytilde_transposed"]

    packed_params["minimizer"] = minimizer
    packed_params["debug"] = debug
    packed_params["verbose"] = verbose
    packed_params["params"] = params
    packed_params["n_threads"] = n_threads
    packed_params["cache_ytilde_transposed"] = cache_ytilde_transposed

    # extract algorithm and use_c_functions from params
    if "algorithm" in params.keys():
        algorithm = params["algorithm"]
        del params["algorithm"]
    else:
        algorithm = ""
    packed_params["algorithm"] = algorithm

    if "use_c_functions" in params.keys():
        use_c_functions = params["use_c_functions"]
        del params["use_c_functions"]
    else:
        use_c_functions = True

    packed_params["use_c_functions"] = use_c_functions

    return packed_params
