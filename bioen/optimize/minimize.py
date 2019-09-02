"""Python common functions
"""
from __future__ import print_function

import os
import sys
import numpy as np

from . import util
from .ext import c_bioen


def set_fast_openmp_flag(flag):
    c_bioen.set_fast_openmp_flag(flag)


def get_fast_openmp_flag():
    return c_bioen.get_fast_openmp_flag()


def show_params(packed_params):
    """
    Function that prints the provided configuration in a readable format.

    Parameters
    ----------
    packed_params: map

    """
    print("minimizer               ", packed_params["minimizer"])
    # print "debug                   ", packed_params["debug"]
    print("verbose                 ", packed_params["verbose"])
    print("params                  ", packed_params["params"])
    print("algorithm               ", packed_params["algorithm"])
    print("use_c_functions         ", packed_params["use_c_functions"])
    print("n_threads               ", packed_params["n_threads"])
    print("cache_ytilde_transposed ", packed_params["cache_ytilde_transposed"])
    print("------------------------------")

    return


# default parameters from template for a given minimizer
def Parameters(minimizer, parameter_mod=""):
    """
    Provides the default parameters for a specific minimizer

    Parameters
    ----------
    minimizer: "lbfgs" "gsl" "scipy"

    Returns
    -------
    params: map structure
    """

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/config/"
    BASE_FILE = "bioen_optimize.yaml"
    TEMPLATE = ROOT_DIR + BASE_FILE

    if not os.path.isfile(TEMPLATE):
        print("Default parameter file (", TEMPLATE, ") cannot be found!")

    packed_params = util.load_template_config_yaml(TEMPLATE, minimizer, parameter_mod)

    return packed_params
