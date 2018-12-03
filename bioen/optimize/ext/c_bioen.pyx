"""Cython interface to the C implementations of the log-weights and the forces methods.
"""

import numpy as np
cimport numpy as np

from libc.setjmp cimport *



cdef extern from "c_bioen_error.h":

    void bioen_manage_error(int,
                            int)
    void _set_ctx(jmp_buf*)


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

    double _opt_bfgs_logw(double*, # g
                          double*, # G
                          double*, # yTIlde
                          double*, # YTilde
                          double*, # w
                          double*, # result
                          double, # theta
                          int, # m
                          int, # n
                          gsl_config_params, # config
                          caching_params, # caching
                          visual_params) # visual

    double _opt_lbfgs_logw(double*, # g
                           double*, # G
                           double*, # yTilde
                           double*, # YTilde
                           double*, # w
                           double*, # result
                           double, # theta
                           int, # m
                           int, # n
                           lbfgs_config_params, # config
                           caching_params, # caching
                           visual_params)  # visual


cdef extern from "c_bioen_kernels_forces.h":

    double _opt_bfgs_forces(double*, # forces
                            double*, # w0
                            double*, # y_param
                            double*, # yTilde
                            double*, # YTilde
                            double*, # result
                            double, # theta
                            int, # m
                            int, # n
                            gsl_config_params, # config
                            caching_params, # caching
                            visual_params) # visual_params visual

    double _opt_lbfgs_forces(double*, # forces
                             double*, # w0
                             double*, # y_param
                             double*, # yTilde
                             double*, # YTilde
                             double*, # result
                             double, # theta
                             int, # m
                             int, # n
                             lbfgs_config_params, # config
                             caching_params, # caching
                             visual_params) # visual

    void _get_weights_from_forces(const double* const, # w0
                                  const double* const, # yTilde
                                  const double* const, # forces
                                  double* const, # w
                                  const int, # caching
                                  const double* const, # yTildeT
                                  double* const, # tmp_n
                                  const size_t, # m
                                  const size_t) # n

    double _bioen_log_posterior_forces(const double* const, # forces
                                       const double* const, # w0
                                       const double* const, # y_param
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

    void   _grad_bioen_log_posterior_forces(const double* const, # forces
                                            const double* const, # w0
                                            const double* const, # y_param
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
    cdef np.ndarray w = np.empty([n], dtype=np.double)

    cdef double weights_sum
    cdef double val

    ## Should not be called into a function
    ## otherwise setjmp will fail

    cdef jmp_buf ctx
    _set_ctx(&ctx)
    error = setjmp(ctx)

    if error == 0:
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

    else:
        raise ValueError("Error bioen_log_posterior_logw " + str(error))
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

    cdef jmp_buf ctx
    _set_ctx(&ctx)
    error = setjmp(ctx)

    if error == 0:
        # 1) compute weights
        weights_sum = _get_weights(<double*> gPrime.data,
                                   <double*> w.data,
                                   <size_t> n)

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
    else:
        raise ValueError("Error grad_bioen_log_posterior_logw " + str(error))


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

    cdef double fmin

    cdef jmp_buf ctx
    _set_ctx(&ctx)
    error = setjmp(ctx)

    if error == 0:
        fmin = _opt_bfgs_logw(<double*> g.data,
                              <double*> G.data,
                              <double*> yTilde.data,
                              <double*> YTilde.data,
                              <double*> w.data,
                              <double*> x.data,
                              <double> theta,
                              <int> m,
                              <int> n,
                              <gsl_config_params> c_params,
                              <caching_params> c_caching_params,
                              <visual_params> c_visual_params)
    else:
        raise ValueError("Error bioen_opt_bfgs_logw " + str(error))

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

    cdef double fmin

    cdef jmp_buf ctx
    _set_ctx(&ctx)
    error = setjmp(ctx)

    if error == 0:
        fmin = _opt_lbfgs_logw(<double*> g.data,
                               <double*> G.data,
                               <double*> yTilde.data,
                               <double*> YTilde.data,
                               <double*> w.data,
                               <double*> x.data,
                               <double>  theta,
                               <int> m,
                               <int> n,
                               <lbfgs_config_params> c_params,
                               <caching_params> c_caching_params,
                               <visual_params> c_visual_params)
    else:
        raise ValueError("Error bioen_opt_lbfgs_logw " + str(error))
    return x, fmin


def bioen_log_posterior_forces(np.ndarray forces,
                               np.ndarray w0,
                               np.ndarray y,
                               np.ndarray yTilde,
                               np.ndarray YTilde,
                               theta,
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

    cdef double val

    cdef jmp_buf ctx
    _set_ctx(&ctx)
    error = setjmp(ctx)

    if error == 0:
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
        val = _bioen_log_posterior_forces(<double*> forces.data,
                                          <double*> w0.data,
                                          <double*> y.data,
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
    else:
        raise ValueError("Error _bioen_log_posterior_forces " + str(error))
    return val


def grad_bioen_log_posterior_forces(np.ndarray forces,
                                    np.ndarray w0,
                                    np.ndarray y,
                                    np.ndarray yTilde,
                                    np.ndarray YTilde,
                                    theta,
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


    cdef jmp_buf ctx
    _set_ctx(&ctx)
    error = setjmp(ctx)

    if error == 0:
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
    else:
        raise ValueError("Error grad_bioen_log_posterior_forces " + str(error))

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

    cdef double fmin

    cdef jmp_buf ctx
    _set_ctx(&ctx)
    error = setjmp(ctx)

    if error == 0:
        fmin = _opt_bfgs_forces(<double*> forces.data,
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

    else:
        raise ValueError("Error bioen_opt_bfgs_forces " + str(error))


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

    cdef double fmin

    cdef jmp_buf ctx
    _set_ctx(&ctx)
    error = setjmp(ctx)

    if error == 0:
        fmin = _opt_lbfgs_forces(<double*> forces.data,
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
    else:
        raise ValueError("Error bioen_opt_lbfgs_forces " + str(error))

    return x, fmin
