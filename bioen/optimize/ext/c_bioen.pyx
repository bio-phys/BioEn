"""Cython interface to the C implementations of the log-weights and the forces methods.
"""

import sys
import time
import numpy as np
cimport numpy as np


cdef extern from "c_bioen_error.h":
    char* bioen_gsl_error(int)
    char* lbfgs_strerror(int)


cdef extern from "c_bioen_kernels_logw.h":
    double _get_weights(const double* const, # g
                        double* const, # w
                        const size_t) # n

    double _bioen_log_posterior_logw(const double* const, # g,
                                     const double* const, # G,
                                     const double* const, # yTilde,
                                     const double* const, # YTilde,
                                     const double* const, # w,
                                     const double* const, # dummy,
                                     const double, # theta,
                                     const int, # caching,
                                     const double* const, # yTildeT,
                                     double*, # tmp_n,
                                     double*, # tmp_m,
                                     const int, # m_int,
                                     const int, # n_int,
                                     const double) # weights_sum

    double _grad_bioen_log_posterior_logw(const double* const, # g,
                                          const double* const, # G,
                                          const double* const, # yTilde,
                                          const double* const, # YTilde,
                                          const double* const, # w,
                                          double* const, # gradient,
                                          const double, # theta,
                                          const int, # caching,
                                          const double* const, # yTildeT,
                                          double*, # tmp_n,
                                          double*, # tmp_m,
                                          const int, # m_int,
                                          const int, # n_int,
                                          const double) # weights_sum)

    double _opt_bfgs_logw(params_t,          # func_params
                          gsl_config_params, # c_params
                          visual_params,     # c_visual_params
                          int *)  # errno

    double _opt_lbfgs_logw(params_t,            #func_params
                           lbfgs_config_params, #c_params
                           visual_params,        #c_visual_params
                           int *) # errno


cdef extern from "c_bioen_kernels_forces.h":
    double _opt_bfgs_forces(params_t, # packed params
                            gsl_config_params, # config
                            visual_params, # visual
                            int *) # errno

    double _opt_lbfgs_forces(params_t, # packed params
                             lbfgs_config_params, # config
                             visual_params, # visual
                             int *) # errno

    void _get_weights_from_forces(const double* const, # w0
                                  const double* const, # yTilde
                                  const double* const, # forces
                                  double* const, # w
                                  const int, # caching
                                  const double* const, # yTildeT
                                  double* const, # tmp_n
                                  const size_t, # m
                                  const size_t) # n

    double _bioen_log_posterior_forces(const double* const, # w0
                                       const double* const, # yTilde
                                       const double* const, # YTilde
                                       const double* const, # w
                                       const double* const, # result
                                       const double,  # theta
                                       const int,  # caching
                                       const double* const,  # yTildeT
                                       double* const,  # tmp_n
                                       double* const,  # tmp_m
                                       const int,  # m_int
                                       const int)  # n_int

    void   _grad_bioen_log_posterior_forces(const double* const, # w0
                                            const double* const, # yTilde
                                            const double* const, # YTilde
                                            const double* const, # w
                                            double* const, # gradient
                                            const double, # theta
                                            const int, # caching
                                            const double* const, # yTildeT
                                            double* const, # tmp_n
                                            double* const, # tmp_m
                                            const int, # m_int
                                            const int) # n_int


# --- GSL status codes, see <gsl_errno.h> ---
# gsl_continue: iteration has not converged
# Though GSL would do more iterations, the result is often OK.
gsl_continue = -2
# gsl_errno: iteration is not making progress towards solution
# Though GSL would do more iterations, the result is often OK.
gsl_enoprog = 27
gsl_success = [gsl_continue, gsl_enoprog, 0]
gsl_continue_msg = "Note: GSL might require more iterations, please check the parameters of the minimizer."

# --- LBFGS status codes, see <lbfgs.h> ---
lbfgs_success = [0, 1, 2]


cdef extern from "c_bioen_common.h":
    int _library_gsl()
    int _library_lbfgs()

    void _omp_set_num_threads(int)
    void _set_fast_openmp_flag(int)
    int _get_fast_openmp_flag()

    struct gsl_config_params  "gsl_config_params":
        double step_size        "step_size"
        double tol              "tol"
        int    max_iterations   "max_iterations"
        int    algorithm        "algorithm"

    struct lbfgs_config_params  "lbfgs_config_params":
        int    linesearch       "linesearch"
        int    max_iterations   "max_iterations"
        double delta            "delta"
        double epsilon          "epsilon"
        double ftol             "ftol"
        double gtol             "gtol"
        double wolfe            "wolfe"
        int    past             "past"
        int    max_linesearch   "max_linesearch"

    struct caching_params  "caching_params":
        int     lcaching        "lcaching"
        double* yTildeT         "yTildeT"
        double* tmp_n           "tmp_n"
        double* tmp_m           "tmp_m"

    struct visual_params  "visual_params":
        size_t debug            "debug"
        size_t verbose          "verbose"

    struct params_t     "params_t":
        double *forces  "forces"
        double *w0      "w0"
        double *g       "g"
        double *G       "G"
        double *yTilde  "yTilde"
        double *YTilde  "YTilde"
        double *w       "w"
        double *result  "result"
        double theta    "theta"
        double *yTildeT "yTildeT"
        int caching     "caching"
        double *tmp_n   "tmp_n"
        double *tmp_m   "tmp_m"
        int m           "m"
        int n           "n"


def set_fast_openmp_flag(flag):
    _set_fast_openmp_flag(flag)


def get_fast_openmp_flag():
    return _get_fast_openmp_flag()


def omp_set_num_threads(i):
    _omp_set_num_threads(i)


def get_gsl_method(algorithm):
    """
    Returns the id of gsl's internal value for algorithm.

    Parameters
    ----------
    string: gsl algorithm

    Returns
    -------
    int: gsl's algorithm id
    """

    if algorithm in ["conjugate_fr", "gsl_multimin_fdfminimizer_conjugate_fr"]:
        return 0
    elif algorithm in ["conjugate_pr", "gsl_multimin_fdfminimizer_conjugate_pr"]:
        return 1
    elif algorithm in ["bfgs2", "gsl_multimin_fdfminimizer_vector_bfgs2"]:
        return 2
    elif algorithm in ["bfgs", "gsl_multimin_fdfminimizer_vector_bfgs"]:
        return 3
    elif algorithm in ["steepest_descent", "gsl_multimin_fdfminimizer_steepest_descent"]:
        return 4
    else:
        raise RuntimeError("{}, GSL return code: {}:{}".format(
            sys._getframe().f_code.co_name, -1, ' The algorithm ' + algorithm + ' is not available.'))


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
    ----------
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
    cdef np.ndarray w = np.empty([n], dtype=np.double)

    cdef double weights_sum
    cdef double val

    # 1) compute weights
    weights_sum = _get_weights(<double*> gPrime.data,
                               <double*> w.data,
                               <size_t> n)
    # 2) compute function
    val = _bioen_log_posterior_logw(<double*> gPrime.data,
                                    <double*> g.data,
                                    <double*> yTilde.data,
                                    <double*> YTilde.data,
                                    <double*> w.data,
                                    <double*> NULL,
                                    <double> theta,
                                    <int> 0,
                                    <double*> NULL,
                                    <double*> tmp_n.data,
                                    <double*> tmp_m.data,
                                    <int> m,
                                    <int> n,
                                    <double> weights_sum)
    return val


def grad_bioen_log_posterior_logw(np.ndarray gPrime, np.ndarray g, np.ndarray G,
                                  np.ndarray yTilde, np.ndarray YTilde, theta,
                                  caching=False, print_timing=False):
    """
    Parameters
    ----------
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

    cdef int use_cache_flag = 0
    cdef np.ndarray yTildeT = np.empty([1], dtype=np.double)
    if caching:
        use_cache_flag = 1
        yTildeT = yTilde.T.copy()

    cdef np.ndarray w = np.empty([n], dtype=np.double)
    cdef np.ndarray tmp_n = np.empty([n], dtype=np.double)
    cdef np.ndarray tmp_m = np.empty([m], dtype=np.double)
    cdef np.ndarray gradient = np.empty([n], dtype=np.double)

    cdef double weights_sum

    if print_timing:
        t0 = time.time()

    # 1) compute weights
    weights_sum = _get_weights(<double*> gPrime.data,
                               <double*> w.data,
                               <size_t> n)

    if print_timing:
        print("_get_weights: {}".format(time.time() - t0))

    # 2) compute function gradient
    _grad_bioen_log_posterior_logw(<double*> gPrime.data,
                                   <double*> G.data,
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
                                   <int> n,
                                   <double> weights_sum)

    if print_timing:
        print("_grad_bioen_log_posterior_logw: {}".format(time.time() - t0))

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
    cdef np.ndarray result = np.empty([n], dtype=np.double)

    # structures containing additional information
    cdef gsl_config_params c_params
    c_params.algorithm      = get_gsl_method(params["algorithm"])
    c_params.tol            = params["params"]["tol"]
    c_params.step_size      = params["params"]["step_size"]
    c_params.max_iterations = params["params"]["max_iterations"]

    cdef visual_params c_visual_params
    c_visual_params.debug   = params["debug"]
    c_visual_params.verbose = params["verbose"]

    cdef params_t  func_params
    func_params.g       = <double*> g.data
    func_params.G       = <double*> G.data
    func_params.yTilde  = <double*> yTilde.data
    func_params.YTilde  = <double*> YTilde.data
    func_params.w       = <double*> w.data
    func_params.result  = <double*> result.data
    func_params.theta   = <double>  theta
    func_params.yTildeT = <double*> yTildeT.data
    func_params.caching = <int>     use_cache_flag
    func_params.tmp_n   = <double*> tmp_n.data
    func_params.tmp_m   = <double*> tmp_m.data
    func_params.m       = <int>     m
    func_params.n       = <int>     n

    cdef int errno = 0

    cdef double fmin = _opt_bfgs_logw(<params_t> func_params,
                                      <gsl_config_params> c_params,
                                      <visual_params> c_visual_params,
                                      <int*> &errno)

    if errno in gsl_success:
        if errno == gsl_continue:
            print(gsl_continue_msg)
        return result, fmin
    else:
        raise RuntimeError("{}, GSL return code: {}:{}".format(
            sys._getframe().f_code.co_name, errno, bioen_gsl_error(errno)))


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
    cdef np.ndarray result = np.empty([n], dtype=np.double)

    cdef lbfgs_config_params c_params
    c_params.linesearch     = params["params"]["linesearch"]
    c_params.max_iterations = params["params"]["max_iterations"]
    c_params.delta          = params["params"]["delta"]
    c_params.epsilon        = params["params"]["epsilon"]
    c_params.ftol           = params["params"]["ftol"]
    c_params.gtol           = params["params"]["gtol"]
    c_params.wolfe          = params["params"]["wolfe"]
    c_params.past           = params["params"]["past"]
    c_params.max_linesearch = params["params"]["max_linesearch"]

    cdef visual_params c_visual_params
    c_visual_params.debug   = params["debug"]
    c_visual_params.verbose = params["verbose"]

    cdef params_t  func_params
    func_params.g       = <double*> g.data
    func_params.G       = <double*> G.data
    func_params.yTilde  = <double*> yTilde.data
    func_params.YTilde  = <double*> YTilde.data
    func_params.w       = <double*> w.data
    func_params.result  = <double*> result.data
    func_params.theta   = <double>  theta
    func_params.yTildeT = <double*> yTildeT.data
    func_params.caching = <int>     use_cache_flag
    func_params.tmp_n   = <double*> tmp_n.data
    func_params.tmp_m   = <double*> tmp_m.data
    func_params.m       = <int>     m
    func_params.n       = <int>     n

    cdef int errno = 0
    cdef double fmin = _opt_lbfgs_logw(<params_t> func_params,
                                       <lbfgs_config_params> c_params,
                                       <visual_params> c_visual_params,
                                       <int*> &errno)

    if errno in lbfgs_success:
        return result, fmin
    else:
        raise RuntimeError("{}, liblbfgs return code: {}:{}".format(
            sys._getframe().f_code.co_name, errno, lbfgs_strerror(errno)))


def bioen_log_posterior_forces(np.ndarray forces,
                               np.ndarray w0,
                               np.ndarray yTilde,
                               np.ndarray YTilde,
                               theta,
                               caching=False):
    """
    Parameters
    ----------
    forcesInit: 1xM matrix
    w0: array of length N
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
    cdef double val
    val = _bioen_log_posterior_forces(<double*> w0.data,
                                      <double*> yTilde.data,
                                      <double*> YTilde.data ,
                                      <double*> w.data,
                                      <double*> NULL,
                                      theta,
                                      <int> 0,
                                      <double*> NULL,
                                      <double*> tmp_n.data,
                                      <double*> tmp_m.data,
                                      <int> m,
                                      <int> n)
    return val


def grad_bioen_log_posterior_forces(np.ndarray forces,
                                    np.ndarray w0,
                                    np.ndarray yTilde,
                                    np.ndarray YTilde,
                                    theta,
                                    caching=False):
    """
    Parameters
    ----------
    forcesInit: 1xM matrix
    w0: array of length N
    yTilde: MxN matrix, M observables y_i / sigma_i for the M structures
    YTilde: 1xM matrix, experimental observables
    theta: float, confidence parameter
    caching: performance optimization; local transposed copy of yTilde (default = False)

    Returns
    -------
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
    _grad_bioen_log_posterior_forces(<double*> w0.data,
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


def bioen_opt_bfgs_forces(np.ndarray forces, np.ndarray w0,
                          np.ndarray yTilde, np.ndarray YTilde, theta, params):
    """
    Parameters
    ----------
    forces: 1xM matrix
    w0: array of length N
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
    cdef np.ndarray w = np.empty([n], dtype=np.double)

    cdef int use_cache_flag = 0
    cdef np.ndarray yTildeT = np.empty([1], dtype=np.double)
    if params["cache_ytilde_transposed"]:
        use_cache_flag = 1
        yTildeT = yTilde.T.copy()

    cdef np.ndarray result = np.empty([m], dtype=np.double)

    cdef gsl_config_params c_params
    c_params.algorithm      = get_gsl_method(params["algorithm"])
    c_params.tol            = params["params"]["tol"]
    c_params.step_size      = params["params"]["step_size"]
    c_params.max_iterations = params["params"]["max_iterations"]

    cdef visual_params c_visual_params
    c_visual_params.debug   = params["debug"]
    c_visual_params.verbose = params["verbose"]

    cdef params_t  func_params
    func_params.forces  = <double*> forces.data
    func_params.w0      = <double*> w0.data
    func_params.yTilde  = <double*> yTilde.data
    func_params.YTilde  = <double*> YTilde.data
    func_params.result  = <double*> result.data
    func_params.w       = <double*> w.data
    func_params.theta   = <double>  theta
    func_params.yTildeT = <double*> yTildeT.data
    func_params.caching = <int>     use_cache_flag
    func_params.tmp_n   = <double*> tmp_n.data
    func_params.tmp_m   = <double*> tmp_m.data
    func_params.m       = <int>     m
    func_params.n       = <int>     n

    cdef int errno = 0

    cdef double fmin = _opt_bfgs_forces(<params_t> func_params,
                                        <gsl_config_params> c_params,
                                        <visual_params> c_visual_params,
                                        <int *> &errno)

    if errno in gsl_success:
        if errno == gsl_continue:
            print(gsl_continue_msg)
        return result, fmin
    else:
        raise RuntimeError("{}, GSL return code: {}:{}".format(
            sys._getframe().f_code.co_name, errno, bioen_gsl_error(errno)))


def bioen_opt_lbfgs_forces(np.ndarray forces, np.ndarray w0,
                           np.ndarray yTilde, np.ndarray YTilde, theta, params):
    """
    Parameters
    ----------
    forces: 1xM matrix
    w0: array of length N
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
    cdef np.ndarray w = np.empty([n], dtype=np.double)

    cdef int use_cache_flag = 0
    cdef np.ndarray yTildeT = np.empty([1], dtype=np.double)
    if params["cache_ytilde_transposed"]:
        use_cache_flag = 1
        yTildeT = yTilde.T.copy()

    cdef np.ndarray result = np.empty([m], dtype=np.double)

    cdef lbfgs_config_params c_conf_params
    c_conf_params.linesearch     = params["params"]["linesearch"]
    c_conf_params.max_iterations = params["params"]["max_iterations"]
    c_conf_params.delta          = params["params"]["delta"]
    c_conf_params.epsilon        = params["params"]["epsilon"]
    c_conf_params.ftol           = params["params"]["ftol"]
    c_conf_params.gtol           = params["params"]["gtol"]
    c_conf_params.wolfe          = params["params"]["wolfe"]
    c_conf_params.past           = params["params"]["past"]
    c_conf_params.max_linesearch = params["params"]["max_linesearch"]

    cdef visual_params c_visual_params
    c_visual_params.debug   = params["debug"]
    c_visual_params.verbose = params["verbose"]

    cdef params_t  func_params
    func_params.forces  = <double*> forces.data
    func_params.w0      = <double*> w0.data
    func_params.yTilde  = <double*> yTilde.data
    func_params.YTilde  = <double*> YTilde.data
    func_params.result  = <double*> result.data
    func_params.w       = <double*> w.data
    func_params.theta   = <double>  theta
    func_params.yTildeT = <double*> yTildeT.data
    func_params.caching = <int>     use_cache_flag
    func_params.tmp_n   = <double*> tmp_n.data
    func_params.tmp_m   = <double*> tmp_m.data
    func_params.m       = <int>     m
    func_params.n       = <int>     n

    cdef int errno = 0

    cdef double fmin = _opt_lbfgs_forces(<params_t> func_params,
                                         <lbfgs_config_params> c_conf_params,
                                         <visual_params> c_visual_params,
                                         <int*> &errno)

    if errno in lbfgs_success:
        return result, fmin
    else:
        raise RuntimeError("{}, liblbfgs return code: {}:{}".format(
            sys._getframe().f_code.co_name, errno, lbfgs_strerror(errno)))
