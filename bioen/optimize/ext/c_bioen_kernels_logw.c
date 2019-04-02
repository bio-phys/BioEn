/** C implementations of the log-weights method. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <stdint.h>

#ifdef ENABLE_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#endif

#ifdef ENABLE_LBFGS
#include <lbfgs.h>
#endif

// #define VERBOSE_DEBUG
#include "c_bioen_common.h"
#include "c_bioen_kernels_logw.h"
#include "c_bioen_error.h"
#include "ompmagic.h"

double _get_weights_sum(const double* const g, double* tmp_n, const size_t n) {
    double s = 0.0;

    if (_fast_openmp_flag) {
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE reduction(+ : s))
            for (size_t j = 0; j < n; ++j) {
                s += exp(g[j]);
            }
        }
    } else {
        // preserve order of summation via scratch array 'tmp_n'
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
                        const double* const G, const double theta, double* const tmp_n,
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
// Note: 'w' needs already be filled with values from a previous call to _get_weights()!
double _bioen_log_posterior_logw(const double* const g, const double* const G,
                                 const double* const yTilde, const double* const YTilde,
                                 const double* const w,
                                 const double* const gradient, // unused
                                 const double theta,
                                 const int caching, // unused
                                 const double* const yTildeT, // unused
                                 double* const tmp_n, double* const tmp_m, const int m_int,
                                 const int n_int, const double weights_sum) {
    const size_t m = (size_t)m_int;
    const size_t n = (size_t)n_int;

    const double val1 = _bioen_log_prior(w, weights_sum, g, G, theta, tmp_n, n);
    const double val2 = _bioen_chi_squared(w, yTilde, YTilde, tmp_m, m, n);

    return (val1 + val2);
}

// Gradient function for the forces method
// Note: 'w' needs already be filled with values from a previous call to _get_weights()!
void _grad_bioen_log_posterior_logw(const double* const g,
                                    const double* const G,
                                    const double* const yTilde,
                                    const double* const YTilde,
                                    const double* const w,
                                    double* const gradient,
                                    const double theta,
                                    const int caching,
                                    const double* const yTildeT,
                                    double* const tmp_n,
                                    double* const tmp_m,
                                    const int m_int,
                                    const int n_int,
                                    const double weights_sum) // unused
                                    {
    const size_t m = (size_t)m_int;
    const size_t n = (size_t)n_int;

    double tmp1 = 0.0;
    double tmp2 = 0.0;

    if (_fast_openmp_flag) {
        //printf("FAST_OPENMP\n");
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR(OMP_SCHEDULE)
            for (size_t i = 0; i < m; i++) {
                double tmp = 0.0;
                PRAGMA_OMP_SIMD(reduction(+ : tmp))
                for (size_t j = 0; j < n; j++) {
                    tmp += yTilde[i * n + j] * w[j];
                }
                tmp_m[i] = tmp;
            }

            if (caching) {
                // use cached transposed yTilde
                PRAGMA_OMP_FOR(OMP_SCHEDULE)
                for (size_t j = 0; j < n; j++) {
                    double tmp = 0.0;
                    PRAGMA_OMP_SIMD(reduction(+ : tmp))
                    for (size_t i = 0; i < m; i++) {
                        tmp += (tmp_m[i] - YTilde[i]) * (yTildeT[j * m + i] - tmp_m[i]);
                    }
                    tmp_n[j] = w[j] * tmp;
                }
            } else {
                PRAGMA_OMP_FOR(OMP_SCHEDULE)
                for (size_t j = 0; j < n; j++) {
                    double tmp = 0.0;
                    for (size_t i = 0; i < m; i++) {
                        tmp += (tmp_m[i] - YTilde[i]) * (yTilde[i * n + j] - tmp_m[i]);
                    }
                    tmp_n[j] = w[j] * tmp;
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
                gradient[j] = w[j] * theta * (g[j] - tmp1 - G[j] + tmp2) + tmp_n[j];
            }
        }  // END PRAGMA_OMP_PARALLEL()
    } else {
        //printf("SLOW_OPENMP\n");
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR(OMP_SCHEDULE)
            for (size_t i = 0; i < m; i++) {
                double tmp = 0.0;
                PRAGMA_OMP_SIMD(reduction(+ : tmp))
                for (size_t j = 0; j < n; j++) {
                    tmp += yTilde[i * n + j] * w[j];
                }
                tmp_m[i] = tmp;
            }

            if (caching) {
                // use cached transposed yTilde
                PRAGMA_OMP_FOR(OMP_SCHEDULE)
                for (size_t j = 0; j < n; j++) {
                    double tmp = 0.0;
                    PRAGMA_OMP_SIMD(reduction(+ : tmp))
                    for (size_t i = 0; i < m; i++) {
                        tmp += (tmp_m[i] - YTilde[i]) * (yTildeT[j * m + i] - tmp_m[i]);
                    }
                    tmp_n[j] = w[j] * tmp;
                }
            } else {
                PRAGMA_OMP_FOR(OMP_SCHEDULE)
                for (size_t j = 0; j < n; j++) {
                    double tmp = 0.0;
                    for (size_t i = 0; i < m; i++) {
                        tmp += (tmp_m[i] - YTilde[i]) * (yTilde[i * n + j] - tmp_m[i]);
                    }
                    tmp_n[j] = w[j] * tmp;
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
                gradient[j] = w[j] * theta * (g[j] - tmp1 - G[j] + tmp2) + tmp_n[j];
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
    double theta = p->theta;
    double* tmp_n = p->tmp_n;
    double* tmp_m = p->tmp_m;
    int m = p->m;
    int n = p->n;

    double* v_ptr = (double*)v->data;

    DEBUG_CHECKPOINT();

    // 1) compute weights
    const double weights_sum = _get_weights(v_ptr, w, (size_t)n);

    // 2) compute function
    double val = _bioen_log_posterior_logw(v_ptr, G, yTilde, YTilde, w, NULL, theta,
                                           -1, NULL, tmp_n, tmp_m, m, n, weights_sum);

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
    double theta = p->theta;
    double* yTildeT = p->yTildeT;
    int caching = p->caching;
    double* tmp_n = p->tmp_n;
    double* tmp_m = p->tmp_m;
    int m = p->m;
    int n = p->n;

    double* v_ptr = (double*)v->data;
    double* result_ptr = (double*)df->data;

    DEBUG_CHECKPOINT();

    // 1) compute weights
    //const double weights_sum = _get_weights(v_ptr, w, (size_t)n);
    _get_weights(v_ptr, w, (size_t)n);

    // 2) compute function gradient
    _grad_bioen_log_posterior_logw(v_ptr, G, yTilde, YTilde, w, result_ptr, theta,
                                   caching, yTildeT, tmp_n, tmp_m, m, n, -1);
}

// GSL interface, for objective and gradient, to evaluate an iteration.
void fdf(const gsl_vector* x, void* params, double* f, gsl_vector* df) {
    params_t* p = (params_t*)params;
    double* G = p->G;
    double* yTilde = p->yTilde;
    double* YTilde = p->YTilde;
    double* w = p->w;
    double theta = p->theta;
    double* yTildeT = p->yTildeT;
    int caching = p->caching;
    double* tmp_n = p->tmp_n;
    double* tmp_m = p->tmp_m;
    int m = p->m;
    int n = p->n;

    double* v_ptr = (double*)x->data;
    double* result_ptr = (double*)df->data;

    DEBUG_CHECKPOINT();

    // 1) compute weights
    const double weights_sum = _get_weights(v_ptr, w, (size_t)n);

    // 2) compute function
    *f = _bioen_log_posterior_logw(v_ptr, G, yTilde, YTilde, w, NULL, theta, caching,
                                   yTildeT, tmp_n, tmp_m, m, n, weights_sum);

    // 3) compute function gradient
    _grad_bioen_log_posterior_logw(v_ptr, G, yTilde, YTilde, w, result_ptr, theta,
                                   caching, yTildeT, tmp_n, tmp_m, m, n, -1);
}
#endif


// GSL optimization interface
double _opt_bfgs_logw(struct params_t func_params,
                      struct gsl_config_params config,
                      struct visual_params visual,
                      int* error) {
    double final_val = 0.;

#if ENABLE_GSL
    *error = 0;

    int n = func_params.n;
    int m = func_params.m;

    int gsl_status;
    gsl_set_error_handler_off();

    if (visual.verbose) {
        printf("\t=========================\n");
        printf("\tcaching_yTilde_tranposed : %s\n",  func_params.caching ? "enabled" : "disabled");
        printf("\tGSL minimizer            : %s\n",
               gsl_multimin_algorithm_names[config.algorithm]);
        printf("\ttol                      : %f\n", config.tol);
        printf("\tstep_size                : %f\n", config.step_size);
        printf("\tmax_iteration            : %d\n", config.max_iterations);
        printf("\t=========================\n");
    }

    double start = get_wtime();

    // Allocate independant variables array
    gsl_vector* x0 = gsl_vector_alloc((size_t)n);

    for (int i = 0; i < n; i++) {
        gsl_vector_set(x0, i, func_params.g[i]);
    }

    // Set up optimizer parameters
    const gsl_multimin_fdfminimizer_type* T = NULL;

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

    gsl_multimin_fdfminimizer* s;
    s = gsl_multimin_fdfminimizer_alloc(T, n);

    gsl_multimin_function_fdf my_func;
    my_func.f = &_bioen_log_posterior_interface;
    my_func.df = &_grad_bioen_log_posterior_interface;
    my_func.fdf = &fdf;
    my_func.n = n;
    my_func.params = &func_params;

    // Initialize the optimizer
    gsl_multimin_fdfminimizer_set(s, &my_func, x0, config.step_size, config.tol);

    DEBUG_PRINT("begin");

    // Main loop
    int iter = 0;
    do {
        if (visual.verbose)
            if ((iter != 0) && ((iter % 1000) == 0))
                printf("\t\titeration %d\n", iter);

        gsl_status = gsl_multimin_fdfminimizer_iterate(s);
        if (gsl_status)
            break;

        gsl_status = gsl_multimin_test_gradient__scipy_optimize_vecnorm(s->gradient, config.tol);

        iter++;
    } while (gsl_status == GSL_CONTINUE && iter < config.max_iterations);

    DEBUG_PRINT("end");

    // Get the final minimizing function parameters
    gsl_vector* x = gsl_multimin_fdfminimizer_x(s);

    // Get minimum value
    final_val = gsl_multimin_fdfminimizer_minimum(s);

    // Copy back the result.
    for (int i = 0; i < n; i++) {
        func_params.result[i] = gsl_vector_get(x, i);
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

    double end = get_wtime();

    gsl_vector_free(x0);
    gsl_multimin_fdfminimizer_free(s);

    // Print profile info
    if (visual.verbose) {
        printf("Optimization terminated successfully\n");
        printf("\tConfig: m=%d and n=%d\n", m, n);
        printf("\tCurrent function value  = %.6lf\n", final_val);
        printf("\tIterations              : %d\n", iter);
        printf("\tMinimization time [s]   : %.12lf\n", end - start);
        printf("\tTime [s] per iteration  : %.12lf\n", (end - start) / iter);
    }

    *error = gsl_status;
#else
    printf("%s\n", message_gsl_unavailable);
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
static lbfgsfloatval_t interface_lbfgs_logw(
    void* instance,
    const lbfgsfloatval_t* new_g,  // current values of func.
    lbfgsfloatval_t* grad_vals,    // current grad values of func.
    const int n, const lbfgsfloatval_t step) {
    params_t* p = (params_t*)instance;
    double* G = p->G;
    double* yTilde = p->yTilde;
    double* YTilde = p->YTilde;
    double* w = p->w;
    double theta = p->theta;
    double* yTildeT = p->yTildeT;
    int caching = p->caching;
    double* tmp_n = p->tmp_n;
    double* tmp_m = p->tmp_m;
    int m = p->m;
    double val = 0.0;

    // pointer aliases for lbfgsfloatval_t input and output types
    double* g_ptr = (double*)new_g;
    double* result_ptr = (double*)grad_vals;

    DEBUG_CHECKPOINT();

    // run get_weights only once
    const double weights_sum = _get_weights(g_ptr, w, (size_t)n);

    // Evaluation of objective function
    val = _bioen_log_posterior_logw(g_ptr, G, yTilde, YTilde, w, NULL, theta, caching,
                                    yTildeT, tmp_n, tmp_m, m, n, weights_sum);

    // Evaluation of gradient
    _grad_bioen_log_posterior_logw(g_ptr, G, yTilde, YTilde, w, result_ptr, theta,
                                   caching, yTildeT, tmp_n, tmp_m, m, n, -1);

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
double _opt_lbfgs_logw(struct params_t func_params,
                       struct lbfgs_config_params config,
                       struct visual_params visual,
                       int* error) {
    double final_result = 0;

#ifdef ENABLE_LBFGS
    *error = 0;

    int m = func_params.m;
    int n = func_params.n;

    if (visual.verbose) {
        printf("L-BFGS minimizer\n");
    }

    lbfgsfloatval_t fx;
    lbfgsfloatval_t* x = lbfgs_malloc(n);
    lbfgs_parameter_t lbfgs_param;

    // Initialize the variables.
    for (int i = 0; i < n; i++) {
        x[i] = func_params.g[i];
    }

    // Initialize the parameters for the L-BFGS optimization.
    lbfgs_parameter_init(&lbfgs_param);

    lbfgs_param.linesearch = config.linesearch;  // default 2
    lbfgs_param.max_iterations = config.max_iterations;
    lbfgs_param.delta = config.delta;      // default: 1e-6
    lbfgs_param.epsilon = config.epsilon;  // default: 1e-4
    lbfgs_param.ftol = config.ftol;        // default: 1e-7
    lbfgs_param.gtol = config.gtol;
    lbfgs_param.wolfe = config.wolfe;
    lbfgs_param.past = config.past;                      // default 10
    lbfgs_param.max_linesearch = config.max_linesearch;  // default: 20

    lbfgs_verbose_logw = visual.verbose;
    if (visual.verbose) {
        printf("\t=========================\n");
        printf("\tcaching_yTilde_tranposed : %s\n",  func_params.caching ? "enabled" : "disabled");
        printf("\tlinesearch               : %d\n", lbfgs_param.linesearch);
        printf("\tmax_iterations           : %d\n", lbfgs_param.max_iterations);
        printf("\tdelta                    : %lf\n", lbfgs_param.delta);
        printf("\tepsilon                  : %lf\n", lbfgs_param.epsilon);
        printf("\tftol                     : %lf\n", lbfgs_param.ftol);
        printf("\tgtol                     : %lf\n", lbfgs_param.gtol);
        printf("\twolfe                    : %lf\n", lbfgs_param.wolfe);
        printf("\tpast                     : %d\n", lbfgs_param.past);
        printf("\tmax_linesearch           : %d\n", lbfgs_param.max_linesearch);
        printf("\t=========================\n");
    }

    iterations_lbfgs_logw = 0;

    DEBUG_PRINT("begin");

    double start = get_wtime();
    int return_value =
        lbfgs(n, x, &fx, interface_lbfgs_logw, progress_logw, &func_params, &lbfgs_param);
    double end = get_wtime();

    DEBUG_PRINT("end");

    if (visual.verbose) {
        printf("\t%s\n", lbfgs_strerror(return_value));
        printf("\tConfig: m=%d and n=%d\n", m, n);
        printf("\tCurrent function value  = %.6lf\n", fx);
        printf("\tIterations              : %zu\n", iterations_lbfgs_logw);
        printf("\tTime(s) of L-BFGS       : %.12lf\n", end - start);
        printf("\tTime(s) per iter        : %.12lf\n", (end - start) / iterations_lbfgs_logw);
    }

    final_result = fx;

    for (int i = 0; i < n; i++) {
        func_params.result[i] = x[i];
    }

    lbfgs_free(x);

    *error = return_value;
#else
    printf("%s\n", message_lbfgs_unavailable);
#endif  // ENABLE_LBFGS

    return final_result;
}
