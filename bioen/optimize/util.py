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

    Raises
    ------
    ZeroDivisionError
        If b is zero.

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

    if (np.size(nonzero_idx) == 0):
        return 0.0, 0

    nonzero_msk = np.zeros(b.shape, dtype=bool)
    nonzero_msk[nonzero_idx] = True
    d_array = np.absolute(a[nonzero_msk] - b[nonzero_msk])/np.absolute(b[nonzero_msk])
    d = np.max(d_array)
    idx = np.where(d_array == d)[0][0]
    return d, idx


def load_template_config_yaml(file_name, minimizer, parameter_mod=""):
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
    # TODO: Simplify the handling of the cfg data structure throughout the optimize module, this is currently highly obfuscated!

    minimizer = minimizer.lower()

    with open(file_name, "r") as fp:
        cfg = yaml.safe_load(fp)

    # modify parameters, overriding the default
    if parameter_mod:
        token_list = parameter_mod.split(',')
        for token in token_list:
            keys, value = token.split('=')
            value = ntype(value)
            keys = keys.split(':')
            nested_set(cfg, keys, value)

    packed_params = {}

    debug = cfg["general"]["debug"]
    verbose = cfg["general"]["verbose"]

    params = {}
    for modif in cfg[minimizer]:
        params[modif] = cfg[minimizer][modif]

    n_threads = cfg["c_functions"]["n_threads"]
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


def ntype(s):
    """Convert a string 's' to its native Python datatype, if possible.
    """
    true_strings = ["true", "t", "yes", "y"]
    false_strings = ["false", "f", "no", "n"]

    def is_bool(s):
        return s.lower() in true_strings + false_strings

    def is_true(s):
        return s.lower() in true_strings

    try:
        r = int(s)
    except Exception:
        try:
            r = float(s)
        except Exception:
            if is_bool(s):
                return is_true(s)
            else:
                return s
        else:
            return r
    else:
        return r


def nested_set(dic, keys, value):
    """Set value in a nested dictionary at the location given by the list of keys.
    """
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value
