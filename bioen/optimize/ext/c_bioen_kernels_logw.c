/** C implementations of the log-weights method. */

#include "c_bioen_kernels_logw.h"
#include "c_bioen_common.h"
#include "ompmagic.h"
#include <stdlib.h>


double _get_weights_sum(const double* const g, double* tmp_n, const size_t n) {
    double s = 0.0;

    if (_fast_openmp_flag) {
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE reduction(+ : s))
            for (size_t j = 0; j < n; ++j) {
                tmp_n[j] = exp(g[j]);
                s += tmp_n[j];
            }
        }
    } else {
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
            for (size_t j = 0; j < n; ++j) {
                tmp_n[j] = exp(g[j]);
            }
        }

        for (size_t j = 0; j < n; ++j) {
            s += tmp_n[j];
        }
    }

    return s;
}


double _get_weights(const double* const g, double* const w, const size_t n) {
    double s = 0.0;

    if (_fast_openmp_flag) {
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE reduction(+ : s))
            for (size_t j = 0; j < n; ++j) {
                w[j] = exp(g[j]);
                s += w[j];
            }

            const double s_inv = 1.0 / s;

            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
            for (size_t j = 0; j < n; ++j) {
                w[j] *= s_inv;
            }
        }
    } else {
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
            for (size_t j = 0; j < n; ++j) {
                w[j] = exp(g[j]);
            }
        }

        for (size_t j = 0; j < n; ++j) {
            s += w[j];
        }
        double s_inv = 1.0 / s;

        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
            for (size_t j = 0; j < n; ++j) {
                w[j] *= s_inv;
            }
        }
    }
    return s;
}


double _bioen_log_prior(const double* const w, const double s, const double* const g,
                        const double* const G, const double theta, double* tmp_n,
                        const size_t n) {
    double val = 0.0;

    if (_fast_openmp_flag) {
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE reduction(+ : val))
            for (size_t j = 0; j < n; j++) {
                tmp_n[j] = (g[j] - G[j]) * w[j];
                val += tmp_n[j];
            }
        }
    } else {
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
            for (size_t j = 0; j < n; j++) {
                tmp_n[j] = (g[j] - G[j]) * w[j];
            }
        }

        for (size_t j = 0; j < n; ++j) {
            val += tmp_n[j];
        }
    }

    const double s0 = _get_weights_sum(G, tmp_n, n);

    val = val - log(s) + log(s0);
    val = val * theta;
    return val;
}


// Objective function for the log_weights method
double _bioen_log_posterior_logw(double* g, double* G, double* yTilde, double* YTilde,
                                 double* w, double* t1, double* t2, double* result,
                                 double theta, int caching, double* yTildeT, double* tmp_n,
                                 double* tmp_m, int m_int, int n_int,
                                 int run_get_weights, double weights_sum) {
    double s;
    size_t m = (size_t)m_int;
    size_t n = (size_t)n_int;

    if (run_get_weights) {
        s = _get_weights(g, w, n);
    } else {
        // 'w' already contains weights from a previous invocation
        s = weights_sum;
    }

    double val1 = _bioen_log_prior(w, s, g, G, theta, tmp_n, n);
    double val2 = _bioen_chi_squared(w, yTilde, YTilde, tmp_m, m, n);
    double val = val1 + val2;

    return val;
}


// Gradient function for the forces method
void _grad_bioen_log_posterior_logw(double* g, double* G, double* yTilde, double* YTilde,
                                    double* w, double* t1, double* t2, double* result,
                                    double theta, int caching, double* yTildeT, double* tmp_n,
                                    double* tmp_m, int m_int, int n_int,
                                    int run_get_weights, double weights_sum) {
    size_t m = (size_t)m_int;
    size_t n = (size_t)n_int;

    if (run_get_weights) {
        _get_weights(g, w, n);
    }

    double tmp1 = 0.0;
    double tmp2 = 0.0;

    if (_fast_openmp_flag) {
        // printf("FAST_OPENMP\n");
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR(OMP_SCHEDULE)
            for (size_t i = 0; i < m; i++) {
                double tmp = 0.0;
                PRAGMA_OMP_SIMD(reduction(+ : tmp))
                for (size_t j = 0; j < n; j++) {
                    tmp += yTilde[i * n + j] * w[j];
                }
                t1[i] = tmp;
            }

            if (caching) {
                // use cached transposed yTilde
                PRAGMA_OMP_FOR(OMP_SCHEDULE)
                for (size_t j = 0; j < n; j++) {
                    double tmp = 0.0;
                    PRAGMA_OMP_SIMD(reduction(+ : tmp))
                    for (size_t i = 0; i < m; i++) {
                        tmp += (t1[i] - YTilde[i]) * (yTildeT[j * m + i] - t1[i]);
                    }
                    t2[j] = w[j] * tmp;
                }
            } else {
                PRAGMA_OMP_FOR(OMP_SCHEDULE)
                for (size_t j = 0; j < n; j++) {
                    double tmp = 0.0;
                    for (size_t i = 0; i < m; i++) {
                        tmp += (t1[i] - YTilde[i]) * (yTilde[i * n + j] - t1[i]);
                    }
                    t2[j] = w[j] * tmp;
                }
            }

            // Potential non-reproducibility due to parallel summation!!!
            PRAGMA_OMP_FOR_SIMD(reduction(+ : tmp1, tmp2))
            for (size_t j = 0; j < n; j++) {
                tmp1 += g[j] * w[j];
                tmp2 += G[j] * w[j];
            }

            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
            for (size_t j = 0; j < n; j++) {
                result[j] = w[j] * theta * (g[j] - tmp1 - G[j] + tmp2) + t2[j];
            }
        }  // END PRAGMA_OMP_PARALLEL()
    } else {
        // printf("SLOW_OPENMP\n");
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR(OMP_SCHEDULE)
            for (size_t i = 0; i < m; i++) {
                double tmp = 0.0;
                PRAGMA_OMP_SIMD(reduction(+ : tmp))
                for (size_t j = 0; j < n; j++) {
                    tmp += yTilde[i * n + j] * w[j];
                }
                t1[i] = tmp;
            }

            if (caching) {
                // use cached transposed yTilde
                PRAGMA_OMP_FOR(OMP_SCHEDULE)
                for (size_t j = 0; j < n; j++) {
                    double tmp = 0.0;
                    PRAGMA_OMP_SIMD(reduction(+ : tmp))
                    for (size_t i = 0; i < m; i++) {
                        tmp += (t1[i] - YTilde[i]) * (yTildeT[j * m + i] - t1[i]);
                    }
                    t2[j] = w[j] * tmp;
                }
            } else {
                PRAGMA_OMP_FOR(OMP_SCHEDULE)
                for (size_t j = 0; j < n; j++) {
                    double tmp = 0.0;
                    for (size_t i = 0; i < m; i++) {
                        tmp += (t1[i] - YTilde[i]) * (yTilde[i * n + j] - t1[i]);
                    }
                    t2[j] = w[j] * tmp;
                }
            }
        }  // END PRAGMA_OMP_PARALLEL()

        // Use a non-parallel version here, do not use any pragma, not even SIMD!!!
        for (size_t j = 0; j < n; j++) {
            tmp1 += g[j] * w[j];
            tmp2 += G[j] * w[j];
        }

        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
            for (size_t j = 0; j < n; j++) {
                result[j] = w[j] * theta * (g[j] - tmp1 - G[j] + tmp2) + t2[j];
            }
        }  // END PRAGMA_OMP_PARALLEL()
    }
}


#ifdef ENABLE_GSL
// GSL interface to evaluate the function.
// The new coordinates are obtained from the gsl_vector v.
// Returns the value for the function evaluation.
double _bioen_log_posterior_interface(const gsl_vector* v, void* params) {
    params_t* p = (params_t*)params;
    double* G = p->G;
    double* yTilde = p->yTilde;
    double* YTilde = p->YTilde;
    double* w = p->w;
    double* t1 = p->t1;
    double* t2 = p->t2;
    // double* result = p->result;
    double theta = p->theta;
    double* yTildeT = p->yTildeT;
    int caching = p->caching;
    double* tmp_n = p->tmp_n;
    double* tmp_m = p->tmp_m;
    int m = p->m;
    int n = p->n;

    double* v_ptr = (double*) v->data;

    // arguments: run get_weights internally
    const int run_get_weights = 1;
    const double weights_sum = -1.0;

    double val = _bioen_log_posterior_logw(v_ptr, G, yTilde, YTilde, w, t1, t2, NULL,
                                           theta, caching, yTildeT, tmp_n, tmp_m, m, n,
                                           run_get_weights, weights_sum);

    return val;
}

// GSL interface to evaluate the gradient
// The new coordinates are obtained from the gsl_vector v.
// Returns an array of coordinates obtain from the gradient function (df)

void _grad_bioen_log_posterior_interface(const gsl_vector* v, void* params, gsl_vector* df) {
    params_t* p = (params_t*)params;
    double* G = p->G;
    double* yTilde = p->yTilde;
    double* YTilde = p->YTilde;
    double* w = p->w;
    double* t1 = p->t1;
    double* t2 = p->t2;
    // double* result = p->result;
    double theta = p->theta;
    double* yTildeT = p->yTildeT;
    int caching = p->caching;
    double* tmp_n = p->tmp_n;
    double* tmp_m = p->tmp_m;
    int m = p->m;
    int n = p->n;

    double* v_ptr = (double*) v->data;
    double* result_ptr = (double*) df->data;

    // arguments: run get_weights internally
    const int run_get_weights = 1;
    const double weights_sum = -1.0;

    _grad_bioen_log_posterior_logw(v_ptr, G, yTilde, YTilde, w, t1, t2, result_ptr,
                                   theta, caching, yTildeT, tmp_n, tmp_m, m, n,
                                   run_get_weights, weights_sum);
}


// GSL interface, for objective and gradient, to evaluate an iteration.
void fdf(const gsl_vector* x, void* params, double* f, gsl_vector* df) {
    params_t* p = (params_t*)params;
    double* G = p->G;
    double* yTilde = p->yTilde;
    double* YTilde = p->YTilde;
    double* w = p->w;
    double* t1 = p->t1;
    double* t2 = p->t2;
    // double* result = p->result;
    double theta = p->theta;
    double* yTildeT = p->yTildeT;
    int caching = p->caching;
    double* tmp_n = p->tmp_n;
    double* tmp_m = p->tmp_m;
    int m = p->m;
    int n = p->n;

    double* v_ptr = (double*) x->data;
    double* result_ptr = (double*) df->data;

    // run get_weights only once per invocation of f and grad f
    const int run_get_weights = 0;
    const double weights_sum = _get_weights(v_ptr, w, (size_t)n);

    *f = _bioen_log_posterior_logw(v_ptr, G, yTilde, YTilde, w, t1, t2, NULL,
                                   theta, caching, yTildeT, tmp_n, tmp_m, m, n,
                                   run_get_weights, weights_sum);

    _grad_bioen_log_posterior_logw(v_ptr, G, yTilde, YTilde, w, t1, t2, result_ptr,
                                   theta, caching, yTildeT, tmp_n, tmp_m, m, n,
                                   run_get_weights, weights_sum);
}
#endif


// GSL optimization interface
double _opt_bfgs_logw(double* g, double* G, double* yTilde, double* YTilde, double* w,
                      double* t1, double* t2, double* result, double theta, int m, int n,
                      struct gsl_config_params config, struct caching_params caching,
                      struct visual_params visual) {
    double final_val = 0.;

#if ENABLE_GSL

    int iter;
    int status1 = 0;
    int status2 = 0;
    params_t* params = NULL;
    int status = 0;

    // Set up arguments in param_t structure
    status += posix_memalign((void**)&params, ALIGN_CACHE, sizeof(params_t));
    params->g = g;
    params->G = G;
    params->yTilde = yTilde;
    params->YTilde = YTilde;
    params->w = w;
    params->t1 = t1;
    params->t2 = t2;
    params->result = result;
    params->theta = theta;
    params->yTildeT = caching.yTildeT;
    params->caching = caching.lcaching;
    params->tmp_n = caching.tmp_n;
    params->tmp_m = caching.tmp_m;
    params->m = m;
    params->n = n;

    if (visual.verbose) {
        printf("\t=========================\n");
        printf("\tcaching_yTilde_tranposed : %s\n", caching.lcaching ? "enabled" : "disabled");
        printf("\tGSL minimizer            : %s\n",
               gsl_multimin_algorithm_names[config.algorithm]);
        printf("\ttol                      : %f\n", config.tol);
        printf("\tstep_size                : %f\n", config.step_size);
        printf("\tmax_iteration            : %zd\n", config.max_iterations);
        printf("\t=========================\n");
    }

    double start = 0;
    double end = 0;
    start = get_wtime();

    // User define error handler.
    gsl_set_error_handler(handler);

    // Allocate independant variables array
    gsl_vector* x0 = NULL;
    x0 = gsl_vector_alloc(n);

    for (size_t i = 0; i < n; i++) {
        gsl_vector_set(x0, i, g[i]);
    }
    // Set up optimizer parameters
    const gsl_multimin_fdfminimizer_type* T = NULL;
    gsl_multimin_fdfminimizer* s;
    gsl_multimin_function_fdf my_func;

    switch (config.algorithm) {
        case (fdfminimizer_conjugate_fr):
            T = gsl_multimin_fdfminimizer_conjugate_fr;
            break;
        case (fdfminimizer_conjugate_pr):
            T = gsl_multimin_fdfminimizer_conjugate_pr;
            break;
        case (fdfminimizer_vector_bfgs):
            T = gsl_multimin_fdfminimizer_vector_bfgs;
            break;
        case (fdfminimizer_vector_bfgs2):
            T = gsl_multimin_fdfminimizer_vector_bfgs2;
            break;
        case (fdfminimizer_steepest_descent):
            T = gsl_multimin_fdfminimizer_steepest_descent;
            break;
    }

    s = gsl_multimin_fdfminimizer_alloc(T, n);
    my_func.f = _bioen_log_posterior_interface;
    my_func.df = _grad_bioen_log_posterior_interface;
    my_func.fdf = fdf;
    my_func.n = n;
    my_func.params = params;

    // Initialize the optimizer
    gsl_multimin_fdfminimizer_set(s, &my_func, x0, config.step_size, config.tol);

    // Main loop
    iter = 0;
    do {
        if (visual.verbose)
            if ((iter != 0) && ((iter % 1000) == 0)) printf("\t\tOpt Iteration %d\n", iter);

        status1 = gsl_multimin_fdfminimizer_iterate(s);
        if (status1 != 0) {
            // if (status1 != GSL_ENOPROG)
            //     gsl_error("At fdfminimizer_iterate",__FILE__,__LINE__,status1
            //     );
            break;
        }

        status2 = gsl_multimin_test_gradient__scipy_optimize_vecnorm(s->gradient, config.tol);

        iter++;

    } while (status2 == GSL_CONTINUE && iter < config.max_iterations);

    // Get the final minimizing function parameters
    const gsl_vector* x = NULL;
    x = gsl_multimin_fdfminimizer_x(s);

    // Get minimum value
    final_val = gsl_multimin_fdfminimizer_minimum(s);

    // Copy back the result.
    for (size_t i = 0; i < n; i++) {
        result[i] = gsl_vector_get(x, i);
    }

    if (visual.verbose) {
        if (iter == 0 || iter == config.max_iterations) {
            printf(
                "    ------------------------------------------------------   "
                "\n");
            printf(
                "    Check the error description to fix the problem           "
                "\n");
            if (iter == 0)
                printf(
                    "    WARNING: Iteration is not making progress            "
                    "\n");
            else if (iter == config.max_iterations)
                printf(
                    "    WARNING: MAX_ITERS reached                           "
                    "\n");
            printf(
                "    ------------------------------------------------------   "
                "\n");
        }
    }

    free(params);
    gsl_vector_free(x0);
    gsl_multimin_fdfminimizer_free(s);

    end = get_wtime();

    // Print profile info
    if (visual.verbose) {
        printf("Optimization terminated successfully\n");
        printf("\tConfig: m=%d and n=%d\n", m, n);
        printf("\tCurrent function value  = %.6lf\n", final_val);
        printf("\tIterations              : %d\n", iter);
        printf("\tTime(s) of BFGS_GSL     : %.12lf\n", end - start);
        printf("\tTime(s) per iter        : %.12lf\n", (end - start) / iter);
    }

#else
    printf("GSL has not been configured properly\n");

#endif

    return final_val;
}



#ifdef ENABLE_LBFGS
// Global variable to count the number of iterations required to converge for
// the libLBFGS version
size_t iterations_lbfgs_logw = 0;
// Global variable to control verbosity for the LibLBFGS version. It is
// controlled through a global variable to avoid increasing parameters on the
// functions.
size_t lbfgs_verbose_logw = 1;

// Interface function used by LibLBFGS to perform an interation.
// The new functions coordinates are provided by LibLBFGS and they are named as
// new_gs.
// Executes function and gradient and returns current function evaluation.
static lbfgsfloatval_t interface_lbfgs_logw(void* instance,
                                            const lbfgsfloatval_t* new_g,  // current values of func.
                                            lbfgsfloatval_t* grad_vals,    // current grad values of func.
                                            const int n,
                                            const lbfgsfloatval_t step) {
    params_t* p = (params_t*)instance;
    double* G = p->G;
    double* yTilde = p->yTilde;
    double* YTilde = p->YTilde;
    double* w = p->w;
    double* t1 = p->t1;
    double* t2 = p->t2;
    //double* result = p->result;
    double theta = p->theta;
    double* yTildeT = p->yTildeT;
    int caching = p->caching;
    double* tmp_n = p->tmp_n;
    double* tmp_m = p->tmp_m;
    int m = p->m;
    // int n          = p->n ;
    double val = 0.0;

    // pointer aliases for lbfgsfloatval_t input and output types
    double* g_ptr = (double*) new_g;
    double* result_ptr = (double*) grad_vals;

    // run get_weights only once
    const int run_get_weights = 0;
    const double weights_sum = _get_weights(g_ptr, w, (size_t)n);

    // Evaluation of objective function
    val = _bioen_log_posterior_logw(g_ptr, G, yTilde, YTilde, w, t1, t2, NULL,
                                    theta, caching, yTildeT, tmp_n, tmp_m, m, n,
                                    run_get_weights, weights_sum);

    // Evaluation of gradient
    _grad_bioen_log_posterior_logw(g_ptr, G, yTilde, YTilde, w, t1, t2, result_ptr,
                                   theta, caching, yTildeT, tmp_n, tmp_m, m, n,
                                   run_get_weights, weights_sum);

    return val;
}


// Function that controls the progress of the LBFGS execution.
// Counts iterations and prints progress after 1000 iterations
static int progress_logw(void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g,
                         const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm,
                         const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k,
                         int ls) {
    iterations_lbfgs_logw++;

    if (lbfgs_verbose_logw)
        if ((iterations_lbfgs_logw != 0) && ((iterations_lbfgs_logw % 1000) == 0))
            printf("\t\tOpt Iteration %zu\n", iterations_lbfgs_logw);

    return 0;
}
#endif  // ENABLE_LBFGS


// LibLBFGS optimization interface

double _opt_lbfgs_logw(double* g, double* G, double* yTilde, double* YTilde, double* w,
                       double* t1, double* t2, double* result, double theta, int m, int n,
                       struct lbfgs_config_params config, struct caching_params caching,
                       struct visual_params visual) {
    double final_result = 0;

#ifdef ENABLE_LBFGS
    int status = 0;
    params_t* params = NULL;

    // Set up arguments in param_t structure
    status += posix_memalign((void**)&params, ALIGN_CACHE, sizeof(params_t));
    params->g = g;
    params->G = G;
    params->yTilde = yTilde;
    params->YTilde = YTilde;
    params->w = w;
    params->t1 = t1;
    params->t2 = t2;
    params->result = result;
    params->theta = theta;
    params->yTildeT = caching.yTildeT;
    params->caching = caching.lcaching;
    params->tmp_n = caching.tmp_n;
    params->tmp_m = caching.tmp_m;
    params->m = m;
    params->n = n;

    if (visual.verbose) {
        printf("L-BFGS minimizer\n");
    }

    double start = 0;
    double end = 0;

    lbfgsfloatval_t fx;
    lbfgsfloatval_t* x = lbfgs_malloc(n);
    lbfgs_parameter_t lbfgs_param;

    if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return 1;
    }

    // Initialize the variables.
    for (size_t i = 0; i < n; i++) {
        x[i] = g[i];
    }

    // Initialize the parameters for the L-BFGS optimization.
    lbfgs_parameter_init(&lbfgs_param);

    lbfgs_param.linesearch = config.linesearch;  // default 2
    lbfgs_param.max_iterations = config.max_iterations;
    lbfgs_param.delta = config.delta;      // default: 1e-6
    lbfgs_param.epsilon = config.epsilon;  // default: 1e-4
    lbfgs_param.ftol = config.ftol;        // default: 1e-7
    lbfgs_param.gtol = config.gtol;
    lbfgs_param.past = config.past;                      // default 10
    lbfgs_param.max_linesearch = config.max_linesearch;  // default: 20

    lbfgs_verbose_logw = visual.verbose;
    if (visual.verbose) {
        printf("\t=========================\n");
        printf("\tcaching_yTilde_tranposed : %s\n", caching.lcaching ? "enabled" : "disabled");
        printf("\tlinesearch               : %d\n", lbfgs_param.linesearch);
        printf("\tmax_iterations           : %d\n", lbfgs_param.max_iterations);
        printf("\tdelta                    : %lf\n", lbfgs_param.delta);
        printf("\tepsilon                  : %lf\n", lbfgs_param.epsilon);
        printf("\tftol                     : %lf\n", lbfgs_param.ftol);
        printf("\tgtol                     : %lf\n", lbfgs_param.gtol);
        printf("\tpast                     : %d\n", lbfgs_param.past);
        printf("\tmax_linesearch           : %d\n", lbfgs_param.max_linesearch);
        printf("\t=========================\n");
    }

    //    Start the L-BFGS optimization; this will invoke the callback functions
    //    evaluate() and progress() when necessary.

    iterations_lbfgs_logw = 0;

    start = get_wtime();
    int return_value =
        lbfgs(n, x, &fx, interface_lbfgs_logw, progress_logw, params, &lbfgs_param);
    end = get_wtime();

    if (visual.verbose) {
        printf("\t%s\n", lbfgs_strerror(return_value));
        printf("\tConfig: m=%d and n=%d\n", m, n);
        printf("\tCurrent function value  = %.6lf\n", fx);
        printf("\tIterations              : %zu\n", iterations_lbfgs_logw);
        printf("\tTime(s) of L-BFGS       : %.12lf\n", end - start);
        printf("\tTime(s) per iter        : %.12lf\n", (end - start) / iterations_lbfgs_logw);
    }

    final_result = fx;

    for (size_t i = 0; i < n; i++) {
        result[i] = x[i];
    }

    lbfgs_free(x);
    free(params);

#else
    printf("LibLBFGS has not been configured properly\n");
#endif // ENABLE_LBFGS

    return final_result;
}
