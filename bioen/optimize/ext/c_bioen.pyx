"""Cython interface to the C implementations of the log-weights and the forces methods.
"""

import numpy as np
cimport numpy as np


cdef extern from "c_bioen_kernels_logw.h":

    double _get_weights(const double* const,
                        double* const,
                        const size_t)

    double _bioen_log_posterior_logw        (   double* , double* , double* ,
                                                double* , double* , double* ,
                                                double* , double* , double  ,
                                                int,
                                                double* , double* , double* ,
                                                int     , int,
                                                double)

    double _grad_bioen_log_posterior_logw   (   double* , double* , double* ,
                                                double* , double* , double* ,
                                                double* , double* , double  ,
                                                int,
                                                double* , double* , double* ,
                                                int     , int,
                                                double)

    double _opt_bfgs_logw                   (   double* , double* , double* ,
                                                double* , double* , double* ,
                                                double* , double* , double  ,
                                                int     , int,
                                                gsl_config_params,
                                                caching_params,
                                                visual_params)

    double _opt_lbfgs_logw                  (   double* , double* , double* ,
                                                double* , double* , double* ,
                                                double* , double* , double  ,
                                                int     , int     ,
                                                lbfgs_config_params,
                                                caching_params,
                                                visual_params)


cdef extern from "c_bioen_kernels_forces.h":

    double _opt_bfgs_forces                 (   double* , double* , double* ,
                                                double* , double* , double* ,
                                                double  , int     , int,
                                                gsl_config_params,
                                                caching_params,
                                                visual_params)

    double _opt_lbfgs_forces                (   double* , double*   , double* ,
                                                double* , double*   , double* ,
                                                double  ,
                                                int     , int,
                                                lbfgs_config_params,
                                                caching_params,
                                                visual_params)

    void   _grad_bioen_log_posterior_forces (   double* , double* , double* ,
                                                double* , double* , double* ,
                                                double*  , double  ,
                                                int     , double* , double* ,
                                                double* , int     , int)

    double _bioen_log_posterior_forces      (   double* , double* , double* ,
                                                double* , double* , double* ,
                                                double*  , double  ,
                                                int     , double* , double* ,
                                                double* , int     , int)

    void _get_weights_from_forces(const double* const,
                                const double* const,
                                const double* const,
                                double* const,
                                const int,
                                const double* const,
                                double* const,
                                const size_t,
                                const size_t)


cdef extern from "c_bioen_common.h":
    int _library_gsl()
    int _library_lbfgs()

    void _set_fast_openmp_flag(int)
    int _get_fast_openmp_flag()

    struct gsl_config_params  "gsl_config_params":
        double step_size        "step_size"
        double tol              "tol"
        size_t max_iterations   "max_iterations"
        size_t algorithm        "algorithm"

    struct lbfgs_config_params  "lbfgs_config_params":
        size_t linesearch       "linesearch"
        size_t max_iterations   "max_iterations"
        double delta            "delta"
        double epsilon          "epsilon"
        double ftol             "ftol"
        double gtol             "gtol"
        int    past             "past"
        int    max_linesearch   "max_linesearch"

    struct caching_params  "caching_params":
        size_t lcaching         "lcaching"
        double* yTildeT         "yTildeT"
        double* tmp_n           "tmp_n"
        double* tmp_m           "tmp_m"

    struct visual_params  "visual_params":
        size_t debug            "debug"
        size_t verbose          "verbose"


def set_fast_openmp_flag(flag):
    _set_fast_openmp_flag(flag)


def get_fast_openmp_flag():
    return _get_fast_openmp_flag()


def get_gsl_method(algorithm):
    """
    Returns the id. of gsl's corresponding internal value to string name algorithm

    Parameters
    ----------
    string: gsl algorithm

    Returns
    -------
    int: gsl's algorithm identificator

    """

    if algorithm == "conjugate_fr" or algorithm == "gsl_multimin_fdfminimizer_conjugate_fr" :
        return 0
    elif algorithm == "conjugate_pr" or algorithm == "gsl_multimin_fdfminimizer_conjugate_pr" :
        return 1
    elif algorithm == "bfgs2" or algorithm == "gsl_multimin_fdfminimizer_vector_bfgs2" :
        return 2
    elif algorithm == "bfgs" or algorithm == "gsl_multimin_fdfminimizer_vector_bfgs" :
        return 3
    elif algorithm == "steepest_descent" or algorithm == "gsl_multimin_fdfminimizer_steepest_descent" :
        return 4
    else:
        # Default is bfgs2
        print("Warning: Algorithm not recognized. Using gsl_multimin_fdfminimizer_vector_bfgs2 as default")
        return 2


def library_gsl ():
    """
    Checks availability of gsl library

    Returns
    -------
    bool: True if available

    """
    if (_library_gsl()):
        return True
    else:
        return False


def library_lbfgs ():
    """
    Checks availability of gsl library

    Returns
    -------
    bool: True if available

    """
    if (_library_lbfgs()):
        return True
    else:
        return False


def bioen_log_posterior_logw(np.ndarray gPrime, np.ndarray g, np.ndarray G,
                             np.ndarray yTilde, np.ndarray YTilde, theta, caching=False):
    """
    Parameters
    ---------
    gPrime: array_like, current log weights
    g: array_like, N log weights (initial log-weights)
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    yTilde: array_like, MxN matrix
    YTilde: array_like, vector with M components
    theta: float, confidence parameter
    caching: performance optimization; local transposed copy of yTilde (default = False)

    Returns
    -------
    double: BioEn loglikelihood
    """
    cdef int m = yTilde.shape[0]
    cdef int n = yTilde.shape[1]

    cdef np.ndarray tmp_n = np.empty([n], dtype=np.double)
    cdef np.ndarray tmp_m = np.empty([m], dtype=np.double)

    # use a dummy array in case caching is not used
    cdef int use_cache_flag = 0
    cdef np.ndarray yTildeT = np.empty([1], dtype=np.double)
    if caching:
        use_cache_flag = 1
        yTildeT = yTilde.T.copy()

    # temporary array for weights
    cdef np.ndarray w = np.empty([n], dtype=np.double)

    # The correspondence of number of parameters between func. and grad. must
    # be preserved when running the optimizer.
    # The temporary arrays t1 and t2 are used on the gradient function.
    # Allocation is done at python level
    cdef np.ndarray t1 = np.empty([m], dtype=np.double)
    cdef np.ndarray t2 = np.empty([n], dtype=np.double)
    # result array
    cdef np.ndarray result = np.empty([n], dtype=np.double)

    # 1) compute weights
    cdef double weights_sum = _get_weights(<double*> gPrime.data,
                                           <double*> w.data,
                                           <size_t> n)

    # 2) compute function
    cdef double val = _bioen_log_posterior_logw(<double*> gPrime.data,
                                                <double*> g.data,
                                                <double*> yTilde.data,
                                                <double*> YTilde.data,
                                                <double*> w.data,
                                                <double*> t1.data,
                                                <double*> t2.data,
                                                <double*> result.data,
                                                <double> theta,
                                                <int> use_cache_flag,
                                                <double*> yTildeT.data,
                                                <double*> tmp_n.data,
                                                <double*> tmp_m.data,
                                                <int> m,
                                                <int> n,
                                                <double> weights_sum)
    return val


def grad_bioen_log_posterior_logw(np.ndarray gPrime, np.ndarray g, np.ndarray G,
                                  np.ndarray yTilde, np.ndarray YTilde, theta, caching=False):
    """
    Parameters
    ---------
    gPrime: array_like, current log weights
    g: array_like, N log weights (initial log-weights)
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    yTilde: array_like, MxN matrix
    YTilde: array_like, vector with M components
    theta: float, confidene parameter
    caching: performance optimization; local transposed copy of yTilde (default = False)

    Returns
    -------
    array_like: gradient
    """
    cdef int m = yTilde.shape[0]
    cdef int n = yTilde.shape[1]

    cdef np.ndarray tmp_n = np.empty([n], dtype=np.double)
    cdef np.ndarray tmp_m = np.empty([m], dtype=np.double)

    cdef int use_cache_flag = 0
    cdef np.ndarray yTildeT = np.empty([1], dtype=np.double)
    if caching:
        use_cache_flag = 1
        yTildeT = yTilde.T.copy()

    # temporary array for weights
    cdef np.ndarray w = np.empty([n], dtype=np.double)
    # temporary arrays
    cdef np.ndarray t1 = np.empty([m], dtype=np.double)
    cdef np.ndarray t2 = np.empty([n], dtype=np.double)
    # gradient array
    cdef np.ndarray gradient = np.empty([n], dtype=np.double)

    # 1) compute weights
    cdef double weights_sum = _get_weights(<double*> gPrime.data,
                                           <double*> w.data,
                                           <size_t> n)

    # 2) compute function gradient
    _grad_bioen_log_posterior_logw(<double*> gPrime.data,
                                   <double*> G.data,
                                   <double*> yTilde.data,
                                   <double*> YTilde.data,
                                   <double*> w.data,
                                   <double*> t1.data,
                                   <double*> t2.data,
                                   <double*> gradient.data,
                                   <double> theta,
                                   <int> use_cache_flag,
                                   <double*> yTildeT.data,
                                   <double*> tmp_n.data,
                                   <double*> tmp_m.data,
                                   <int> m,
                                   <int> n,
                                   <double> weights_sum)
    return gradient


def bioen_opt_bfgs_logw(np.ndarray g,
                        np.ndarray G,
                        np.ndarray yTilde,
                        np.ndarray YTilde,
                        theta,
                        params):
    """
    Parameters
    ---------
    g: array_like, current log weights
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    yTilde: array_like, MxN matrix
    YTilde: array_like, vector with M components
    theta: float, confidene parameter
    params: set of configuration parameters

    Returns
    -------
    xfinal: array_like, minimizing function parameters
    fmin: float, minimum
    """
    cdef int m = yTilde.shape[0]
    cdef int n = yTilde.shape[1]

    cdef np.ndarray tmp_n = np.empty([n], dtype=np.double)
    cdef np.ndarray tmp_m = np.empty([m], dtype=np.double)

    cdef int use_cache_flag = 0
    cdef np.ndarray yTildeT = np.empty([1], dtype=np.double)
    if params["cache_ytilde_transposed"]:
        use_cache_flag = 1
        yTildeT = yTilde.T.copy()

    # temporary array for the weights
    cdef np.ndarray w = np.empty([n], dtype=np.double)
    # temporary arrays
    cdef np.ndarray t1 = np.empty([m], dtype=np.double)
    cdef np.ndarray t2 = np.empty([n], dtype=np.double)
    # x array
    cdef np.ndarray x = np.empty([n], dtype=np.double)

    # structures containing additional information
    cdef gsl_config_params c_params
    cdef caching_params c_caching_params
    cdef visual_params c_visual_params

    c_params.algorithm      = get_gsl_method(params["algorithm"])
    c_params.tol            = params["params"]["tol"]
    c_params.step_size      = params["params"]["step_size"]
    c_params.max_iterations = params["params"]["max_iterations"]

    c_caching_params.lcaching   = use_cache_flag
    c_caching_params.yTildeT    = <double*> yTildeT.data
    c_caching_params.tmp_n      = <double*> tmp_n.data
    c_caching_params.tmp_m      = <double*> tmp_m.data

    c_visual_params.debug   = params["debug"]
    c_visual_params.verbose = params["verbose"]

    cdef double fmin = _opt_bfgs_logw(<double*> g.data,
                                      <double*> G.data,
                                      <double*> yTilde.data,
                                      <double*> YTilde.data,
                                      <double*> w.data,
                                      <double*> t1.data,
                                      <double*> t2.data,
                                      <double*> x.data,
                                      <double> theta,
                                      <int> m,
                                      <int> n,
                                      <gsl_config_params> c_params,
                                      <caching_params> c_caching_params,
                                      <visual_params> c_visual_params)
    return x, fmin


def bioen_opt_lbfgs_logw(np.ndarray g,
                         np.ndarray G,
                         np.ndarray yTilde,
                         np.ndarray YTilde,
                         theta,
                         params):
    """
    Parameters
    ---------
    g: array_like, current log weights
    G: array_like, vector with N components, derived from BioEn inital weights (reference probabilities)
    yTilde: array_like, MxN matrix
    YTilde: array_like, vector with M components
    theta: float, confidene parameter
    params: set of configuration parameters

    Returns
    -------
    xfinal: array_like, miniming function parameters
    fmin: float, minimum
    -------
    """
    cdef int m = yTilde.shape[0]
    cdef int n = yTilde.shape[1]

    cdef np.ndarray tmp_n = np.empty([n], dtype=np.double)
    cdef np.ndarray tmp_m = np.empty([m], dtype=np.double)

    cdef int use_cache_flag = 0
    cdef np.ndarray yTildeT = np.empty([1], dtype=np.double)
    if params["cache_ytilde_transposed"]:
        use_cache_flag = 1
        yTildeT = yTilde.T.copy()

    # temporary array for the weights
    cdef np.ndarray w = np.empty([n], dtype=np.double)
    # temporary arrays
    cdef np.ndarray t1 = np.empty([m], dtype=np.double)
    cdef np.ndarray t2 = np.empty([n], dtype=np.double)
    # x array
    cdef np.ndarray x = np.empty([n], dtype=np.double)

    cdef lbfgs_config_params c_params
    cdef caching_params c_caching_params
    cdef visual_params c_visual_params

    c_params.linesearch     = params["params"]["linesearch"]
    c_params.max_iterations = params["params"]["max_iterations"]
    c_params.delta          = params["params"]["delta"]
    c_params.epsilon        = params["params"]["epsilon"]
    c_params.ftol           = params["params"]["ftol"]
    c_params.gtol           = params["params"]["gtol"]
    c_params.past           = params["params"]["past"]
    c_params.max_linesearch = params["params"]["max_linesearch"]

    c_caching_params.lcaching   = use_cache_flag
    c_caching_params.yTildeT    = <double*> yTildeT.data
    c_caching_params.tmp_n      = <double*> tmp_n.data
    c_caching_params.tmp_m      = <double*> tmp_m.data

    c_visual_params.debug   = params["debug"]
    c_visual_params.verbose = params["verbose"]

    cdef double fmin = _opt_lbfgs_logw(<double*> g.data,
                                       <double*> G.data,
                                       <double*> yTilde.data,
                                       <double*> YTilde.data,
                                       <double*> w.data,
                                       <double*> t1.data,
                                       <double*> t2.data,
                                       <double*> x.data,
                                       <double>  theta,
                                       <int> m,
                                       <int> n,
                                       <lbfgs_config_params> c_params,
                                       <caching_params> c_caching_params,
                                       <visual_params> c_visual_params)
    return x, fmin


def bioen_log_posterior_forces(np.ndarray forces, np.ndarray w0,
                               np.ndarray y,      np.ndarray yTilde,
                               np.ndarray YTilde, theta,
                               caching=False):
    """
    Parameters
    ----------
    forcesInit: 1xM matrix
    w0: array of length N
    y: MxN matrix, M observables calculate for the M structures
    yTilde: MxN matrix, M observables y_i / sigma_i for the M structures
    YTilde: 1xM matrix, experimental observables
    theta: float, confidence parameter
    caching: performance optimization; local transposed copy of yTilde (default = False)

    Returns
    -------
    min: float
    """
    cdef int m = yTilde.shape[0]
    cdef int n = yTilde.shape[1]

    cdef np.ndarray tmp_n = np.empty([n], dtype=np.double)
    cdef np.ndarray tmp_m = np.empty([m], dtype=np.double)
    cdef np.ndarray w = np.empty([n], dtype=np.double)

    cdef int use_cache_flag = 0
    cdef np.ndarray yTildeT = np.empty([1], dtype=np.double)
    if caching:
        use_cache_flag = 1
        yTildeT = yTilde.T.copy()

    # 1) compute weights
    _get_weights_from_forces(<double*> w0.data,
                             <double*> yTilde.data,
                             <double*> forces.data,
                             <double*> w.data,
                             <int> use_cache_flag,
                             <double*> yTildeT.data,
                             <double*> tmp_n.data,
                             <size_t> m,
                             <size_t> n)

    # 2) compute function
    cdef double val = _bioen_log_posterior_forces(<double*> forces.data,
                                                  <double*> w0.data,
                                                  <double*> y.data,
                                                  <double*> yTilde.data,
                                                  <double*> YTilde.data ,
                                                  <double*> w.data,
                                                  NULL,
                                                  theta,
                                                  <int> use_cache_flag,
                                                  <double*> yTildeT.data,
                                                  <double*> tmp_n.data,
                                                  <double*>tmp_m.data,
                                                  <int> m,
                                                  <int> n)
    return  val


def grad_bioen_log_posterior_forces(np.ndarray forces, np.ndarray w0,
                                    np.ndarray y,      np.ndarray yTilde,
                                    np.ndarray YTilde, theta,
                                    caching=False):
    """
    Parameters
    ----------
    forcesInit: 1xM matrix
    w0: array of length N
    y: MxN matrix, M observables calculate for the M structures
    yTilde: MxN matrix, M observables y_i / sigma_i for the M structures
    YTilde: 1xM matrix, experimental observables
    theta: float, confidence parameter
    caching: performance optimization; local transposed copy of yTilde (default = False)

    Returns:
    --------
    array_like: gradient
    """
    cdef int m = yTilde.shape[0]
    cdef int n = yTilde.shape[1]

    cdef np.ndarray tmp_n = np.empty([n], dtype=np.double)
    cdef np.ndarray tmp_m = np.empty([m], dtype=np.double)
    cdef np.ndarray w = np.empty([n], dtype=np.double)

    cdef np.ndarray yTildeT = np.empty([1], dtype=np.double)
    cdef int use_cache_flag = 0
    if caching:
        use_cache_flag = 1
        yTildeT = yTilde.T.copy()

    cdef np.ndarray gradient = np.empty([m], dtype=np.double)

    # 1) compute weights
    _get_weights_from_forces(<double*> w0.data,
                             <double*> yTilde.data,
                             <double*> forces.data,
                             <double*> w.data,
                             <int> use_cache_flag,
                             <double*> yTildeT.data,
                             <double*> tmp_n.data,
                             <size_t> m,
                             <size_t> n)

    # 2) compute function gradient
    _grad_bioen_log_posterior_forces(<double*> forces.data,
                                     <double*> w0.data,
                                     <double*> y.data,
                                     <double*> yTilde.data,
                                     <double*> YTilde.data,
                                     <double*> w.data,
                                     <double*> gradient.data,
                                     <double> theta,
                                     <int> use_cache_flag,
                                     <double*> yTildeT.data,
                                     <double*> tmp_n.data,
                                     <double*> tmp_m.data,
                                     <int> m,
                                     <int> n)
    return gradient


def bioen_opt_bfgs_forces(np.ndarray forces,  np.ndarray w0,      np.ndarray y_param,
                          np.ndarray yTilde,  np.ndarray YTilde,  theta, params):
    """
    Parameters
    ----------
    forces: 1xM matrix
    w0: array of length N
    y: MxN matrix, M observables calculate for the M structures
    yTilde: MxN matrix, M observables y_i / sigma_i for the M structures
    YTilde: 1xM matrix, experimental observables
    theta: float, confidence parameter
    params: set of configuration parameters

    Returns
    -------
    x: array_like, minimizing function parameters
    fmin: float, minimum
    """
    cdef int m = yTilde.shape[0]
    cdef int n = yTilde.shape[1]

    cdef np.ndarray tmp_n = np.empty([n], dtype=np.double)
    cdef np.ndarray tmp_m = np.empty([m], dtype=np.double)

    cdef int use_cache_flag = 0
    cdef np.ndarray yTildeT = np.empty([1], dtype=np.double)
    if params["cache_ytilde_transposed"]:
        use_cache_flag = 1
        yTildeT = yTilde.T.copy()

    cdef np.ndarray x = np.empty([m], dtype=np.double)

    cdef gsl_config_params c_params
    cdef caching_params c_caching_params
    cdef visual_params c_visual_params

    c_params.algorithm      = get_gsl_method(params["algorithm"])
    c_params.tol            = params["params"]["tol"]
    c_params.step_size      = params["params"]["step_size"]
    c_params.max_iterations = params["params"]["max_iterations"]

    c_caching_params.lcaching = use_cache_flag
    c_caching_params.yTildeT  = <double*> yTildeT.data
    c_caching_params.tmp_n    = <double*> tmp_n.data
    c_caching_params.tmp_m    = <double*> tmp_m.data

    c_visual_params.debug   = params["debug"]
    c_visual_params.verbose = params["verbose"]

    cdef double fmin = _opt_bfgs_forces(<double*> forces.data,
                                        <double*> w0.data,
                                        <double*> y_param.data,
                                        <double*> yTilde.data,
                                        <double*> YTilde.data,
                                        <double*> x.data,
                                        <double> theta,
                                        <int> m,
                                        <int> n,
                                        <gsl_config_params> c_params,
                                        <caching_params> c_caching_params,
                                        <visual_params> c_visual_params)
    return x, fmin


def bioen_opt_lbfgs_forces(np.ndarray forces, np.ndarray w0,     np.ndarray y_param,
                           np.ndarray yTilde, np.ndarray YTilde, theta, params):
    """
    Parameters
    ----------
    forces: 1xM matrix
    w0: array of length N
    y: MxN matrix, M observables calculate for the M structures
    yTilde: MxN matrix, M observables y_i / sigma_i for the M structures
    YTilde: 1xM matrix, experimental observables
    theta: float, confidence parameter
    params: set of configuration parameters

    Returns
    -------
    x: array_like, minimizing function parameters
    fmin: float, minimum
    """
    cdef int m = yTilde.shape[0]
    cdef int n = yTilde.shape[1]

    cdef np.ndarray tmp_n = np.empty([n], dtype=np.double)
    cdef np.ndarray tmp_m = np.empty([m], dtype=np.double)

    cdef int use_cache_flag = 0
    cdef np.ndarray yTildeT = np.empty([1], dtype=np.double)
    if params["cache_ytilde_transposed"]:
        use_cache_flag = 1
        yTildeT = yTilde.T.copy()

    cdef np.ndarray x = np.empty([m], dtype=np.double)

    cdef lbfgs_config_params c_params
    cdef caching_params c_caching_params
    cdef visual_params c_visual_params

    c_params.linesearch     = params["params"]["linesearch"]
    c_params.max_iterations = params["params"]["max_iterations"]
    c_params.delta          = params["params"]["delta"]
    c_params.epsilon        = params["params"]["epsilon"]
    c_params.ftol           = params["params"]["ftol"]
    c_params.gtol           = params["params"]["gtol"]
    c_params.past           = params["params"]["past"]
    c_params.max_linesearch = params["params"]["max_linesearch"]

    c_caching_params.lcaching = use_cache_flag
    c_caching_params.yTildeT  = <double*> yTildeT.data
    c_caching_params.tmp_n    = <double*> tmp_n.data
    c_caching_params.tmp_m    = <double*> tmp_m.data

    c_visual_params.debug   = params["debug"]
    c_visual_params.verbose = params["verbose"]

    cdef double fmin = _opt_lbfgs_forces(<double*> forces.data,
                                         <double*> w0.data,
                                         <double*> y_param.data,
                                         <double*> yTilde.data,
                                         <double*> YTilde.data,
                                         <double*> x.data,
                                         <double> theta,
                                         <int> m,
                                         <int> n,
                                         <lbfgs_config_params> c_params,
                                         <caching_params> c_caching_params,
                                         <visual_params> c_visual_params)
    return x, fmin
