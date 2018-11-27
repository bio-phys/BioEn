/** C implementations of (grad_)_log_posterior(_forces).
 */

#include "c_bioen_kernels_forces.h"
#include <float.h>
#include <stdlib.h>
#include "c_bioen_common.h"
#include "ompmagic.h"

// Alias for minimum double value
const double dmin = DBL_MIN;
// Alias for maximum double value
const double dmax = DBL_MAX;
const double dmax_neg = -DBL_MAX;

size_t iterations_lbfgs_forces = 0;
// Global variable to control verbosity for the LibLBFGS version. It is
// controlled through a global variable to avoid increasing parameters on the
// functions.
size_t lbfgs_verbose_forces = 1;


#ifdef ENABLE_LBFGS
// Interface function used by LibLBFGS to perform an interation.
// The new functions coordinates are provided by LibLBFGS and they are named as
// new_forces.
// Executes function and gradient and returns current function evaluation.
static lbfgsfloatval_t interface_lbfgs_forces(
    void* instance,
    const lbfgsfloatval_t* new_forces,  // current values of func.
    lbfgsfloatval_t* grad_vals,         // current grad values of func.
    const int num_vars, const lbfgsfloatval_t step) {
    params_t* p = (params_t*)instance;
    double* w0 = p->w0;
    double* w = p->w;
    double* y_param = p->y_param;
    double* yTilde = p->yTilde;
    double* YTilde = p->YTilde;
    double* result = p->result;
    double theta = p->theta;
    double* yTildeT = p->yTildeT;
    int caching = p->caching;
    double* tmp_n = p->tmp_n;
    double* tmp_m = p->tmp_m;
    int m = p->m;
    int n = p->n;
    double ret_result = 0.0;

    // Evaluation of objective function
    ret_result =
        _bioen_log_posterior_forces((double*)new_forces, w0, y_param, yTilde, YTilde, w, result,
                                    theta, caching, yTildeT, tmp_n, tmp_m, m, n);

    // Evaluation of gradient
    _grad_bioen_log_posterior_forces((double*)new_forces, w0, y_param, yTilde, YTilde, w, result,
                                     theta, caching, yTildeT, tmp_n, tmp_m, m, n);

    // fetch new gradient
    for (size_t i = 0; i < m; i++) {
        grad_vals[i] = result[i];
    }

    return ret_result;
}

static int progress_forces(void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g,
                           const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm,
                           const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n,
                           int k, int ls) {
    iterations_lbfgs_forces++;

    if (lbfgs_verbose_forces)
        if ((iterations_lbfgs_forces != 0) && ((iterations_lbfgs_forces % 1000) == 0))
            printf("\t\tOpt Iteration %zu\n", iterations_lbfgs_forces);

    return 0;
}
#endif


/**
 * Maximum function for OpenMP reduction.
 */
double maximum(double a, double b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}
// Declare our custom maximum reduction because this was only added in OpenMP 3.1 (2011).
// #pragma omp declare reduction(maximum : double : omp_out=maximum(omp_out, omp_in)) initializer(omp_priv=dmax_neg)


// Calculates the average
// # --- dimension exploration ---
// IN:  w         [ n ]
// IN:  yTilde    [ m * n ]
// OUT: yTildeAve [ m ]
void _getAve(double* w, double* yTilde, double* yTildeAve, size_t m, size_t n) {
    // IN:  w         [ n ]
    // IN:  yTilde    [ m * n ]
    // OUT: yTildeAve [ m ]
    PRAGMA_OMP_PARALLEL(default(shared))
    {
        PRAGMA_OMP_FOR(OMP_SCHEDULE)
        for (size_t i = 0; i < m; i++) {
            double tmp = 0.0;
            PRAGMA_OMP_SIMD(reduction(+ : tmp))
            for (size_t j = 0; j < n; j++) {
                tmp += yTilde[i * n + j] * w[j];
            }
            yTildeAve[i] = tmp;
        }
    }
    return;
}

// Get weights from forces
// IN:  w0:     [Nx1]
// IN:  forces: [1xM]
// IN:  yTilde: [MxN]
// OUT: w       [N]

void _get_weights_from_forces(double* w0, double* yTilde, double* forces, double* w,
                              int caching, double* yTildeT, double* tmp_n, size_t m, size_t n) {
    if (_fast_openmp_flag) {
        double x_max = -dmax;
        double s = 0.0;
        PRAGMA_OMP_PARALLEL(default(shared))
        {
            if (caching) {
                // use cached transposed yTilde
                PRAGMA_OMP_FOR(OMP_SCHEDULE)
                for (size_t j = 0; j < n; j++) {
                    double tmp = 0.0;
                    PRAGMA_OMP_SIMD(reduction(+ : tmp))
                    for (size_t i = 0; i < m; i++) {
                        tmp += forces[i] * yTildeT[j * m + i];
                    }
                    tmp_n[j] = tmp;
                }
            } else {
                PRAGMA_OMP_FOR(OMP_SCHEDULE)
                for (size_t j = 0; j < n; j++) {
                    double tmp = 0.0;
                    for (size_t i = 0; i < m; i++) {
                        tmp += forces[i] * yTilde[i * n + j];
                    }
                    tmp_n[j] = tmp;
                }
            }

            // double x_max = tmp_n[0];
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE reduction(max : x_max))
            for (size_t j = 0; j < n; j++) {
                x_max = x_max > tmp_n[j] ? x_max : tmp_n[j];
            }

            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
            for (size_t j = 0; j < n; j++) {
                w[j] = w0[j] * exp(tmp_n[j] - x_max);
            }

            // the following parallel block potentially leads to non-reproducibility!
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE reduction(+ : s))
            for (size_t j = 0; j < n; j++) {
                s += w[j];
            }
            const double s_inv = 1.0 / s;

            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
            for (size_t j = 0; j < n; j++) {
                w[j] = s_inv * w[j];
            }
        }
    } else {
        double x_max = -dmax;
        PRAGMA_OMP_PARALLEL(default(shared))
        {
            if (caching) {
                // use cached transposed yTilde
                PRAGMA_OMP_FOR(OMP_SCHEDULE)
                for (size_t j = 0; j < n; j++) {
                    double tmp = 0.0;
                    PRAGMA_OMP_SIMD(reduction(+ : tmp))
                    for (size_t i = 0; i < m; i++) {
                        tmp += forces[i] * yTildeT[j * m + i];
                    }
                    tmp_n[j] = tmp;
                }
            } else {
                PRAGMA_OMP_FOR(OMP_SCHEDULE)
                for (size_t j = 0; j < n; j++) {
                    double tmp = 0.0;
                    for (size_t i = 0; i < m; i++) {
                        tmp += forces[i] * yTilde[i * n + j];
                    }
                    tmp_n[j] = tmp;
                }
            }

            // double x_max = tmp_n[0];
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE reduction(max : x_max))
            for (size_t j = 0; j < n; j++) {
                x_max = x_max > tmp_n[j] ? x_max : tmp_n[j];
            }

            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
            for (size_t j = 0; j < n; j++) {
                w[j] = w0[j] * exp(tmp_n[j] - x_max);
            }
        }

        // Do _not_ parallelize the following block because it leads to non-reproducibility!
        double s = 0.0;
        for (size_t j = 0; j < n; j++) {
            s += w[j];
        }
        const double s_inv = 1.0 / s;

        PRAGMA_OMP_PARALLEL(default(shared))
        {
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
            for (size_t j = 0; j < n; j++) {
                w[j] = s_inv * w[j];
            }
        }
    }
    return;
}

// Objective function for the forces method
// Py:   w=get_weights_from_forces_original(w0, yTilde, forces)
// Py:   chiSqr = chiSqrTerm_original(w, yTilde, YTilde)
// Py:   #   selecting non-zero weights because lim_{w->0} w log(w) = 0
// Py:   ind=np.where(w>0)[0]
// Py:   tmp = theta*np.dot((np.log(w[ind]/w0[ind])).T, w[ind])[0,0]+chiSqr

double _bioen_log_posterior_forces(double* forces, double* w0, double* y_param, double* yTilde,
                                   double* YTilde, double* w, double* result, double theta, int caching,
                                   double* yTildeT, double* tmp_n, double* tmp_m, int m_int,
                                   int n_int) {

    double chiSqr = 0.0;
    double d = 0.0;
    size_t m = (size_t)m_int;
    size_t n = (size_t)n_int;

    chiSqr = _bioen_chi_squared(w, yTilde, YTilde, tmp_m, m, n);

    if (_fast_openmp_flag) {
        PRAGMA_OMP_PARALLEL(default(shared))
        {
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE reduction(+ : d))
            for (size_t j = 0; j < n; j++) {
                if ((w[j] >= dmin) && (w0[j] >= dmin)) {
                    tmp_n[j] = (log(w[j]) - log(w0[j])) * w[j];
                } else {
                    tmp_n[j] = 0.0;
                }

                d += tmp_n[j];
            }
        }
    } else {
        PRAGMA_OMP_PARALLEL(default(shared))
        {
            PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
            for (size_t j = 0; j < n; j++) {
                if ((w[j] >= dmin) && (w0[j] >= dmin)) {
                    tmp_n[j] = (log(w[j]) - log(w0[j])) * w[j];
                } else {
                    tmp_n[j] = 0.0;
                }
            }
        }

        for (size_t j = 0; j < n; j++) {
            d += tmp_n[j];
        }
    }

    return d * theta + chiSqr;
}

// Gradient function for the forces method
// OUT:  result  [m]

void _grad_bioen_log_posterior_forces(double* forces, double* w0, double* y_param,
                                      double* yTilde, double* YTilde, double* w, double* result,
                                      double theta, int caching, double* yTildeT, double* tmp_n,
                                      double* tmp_m, int m_int, int n_int) {
    size_t m = (size_t)m_int;
    size_t n = (size_t)n_int;

    _getAve(w, yTilde, tmp_m, m, n);

    PRAGMA_OMP_PARALLEL(default(shared))
    {
        if (caching) {
            // version using yTildeT
            PRAGMA_OMP_FOR(OMP_SCHEDULE)
            for (size_t j = 0; j < n; j++) {
                double d = 0.0;
                PRAGMA_OMP_SIMD(reduction(+ : d))
                for (size_t i = 0; i < m; i++) {
                    d += yTildeT[j * m + i] * (tmp_m[i] - YTilde[i]);
                }
                tmp_n[j] = d;
            }
        } else {
            // version without yTildeT
            PRAGMA_OMP_FOR(OMP_SCHEDULE)
            for (size_t j = 0; j < n; j++) {
                double d = 0.0;
                for (size_t i = 0; i < m; i++) {
                    d += yTilde[i * n + j] * (tmp_m[i] - YTilde[i]);
                }
                tmp_n[j] = d;
            }
        }

        PRAGMA_OMP_FOR_SIMD(OMP_SCHEDULE)
        for (size_t j = 0; j < n; j++) {
            double d = 1.0;
            if ((w[j] >= dmin) && (w0[j] >= dmin)) {
                d += log(w[j]) - log(w0[j]);
            }
            tmp_n[j] = (d * theta + tmp_n[j]) * w[j];
        }

        PRAGMA_OMP_FOR(OMP_SCHEDULE)
        for (size_t i = 0; i < m; i++) {
            double d = 0.0;
            PRAGMA_OMP_SIMD(reduction(+ : d))
            for (size_t j = 0; j < n; j++) {
                d += (yTilde[i * n + j] - tmp_m[i]) * tmp_n[j];
            }
            result[i] = d;
        }
    }

    return;
}


#ifdef ENABLE_GSL

// GSL interface to evaluate the function.
// The new coordinates are obtained from the gsl_vector v.
// Returns the value for the function evaluation.
double _bioen_log_posterior_forces_interface(const gsl_vector* v, void* params) {
    params_t* p = (params_t*)params;

    // double *forces = p->forces;
    double* w0 = p->w0;
    double* y_param = p->y_param;
    double* yTilde = p->yTilde;
    double* YTilde = p->YTilde;
    double* w = p->w;
    double theta = p->theta;
    int caching = p->caching;
    double* yTildeT = p->yTildeT;
    double* tmp_n = p->tmp_n;
    double* tmp_m = p->tmp_m;
    int m = p->m;
    int n = p->n;

    double ret_result = 0.0;
    double* v_ptr = (double*) v->data;

    _get_weights_from_forces (w0, yTilde, v_ptr, w, caching, yTildeT, tmp_n, m, n);

    ret_result = _bioen_log_posterior_forces(v_ptr, w0, y_param, yTilde, YTilde, w, NULL,
                                             theta, caching, yTildeT, tmp_n, tmp_m, m, n);

    return (ret_result);
}


// GSL interface to evaluate the gradient
// The new coordinates are obtained from the gsl_vector v.
// Returns an array of coordinates obtain from the gradient function (df)
void _grad_bioen_log_posterior_forces_interface(const gsl_vector* v, void* params,
                                                gsl_vector* df) {
    params_t* p = (params_t*)params;

    double* w0 = p->w0;
    double* y_param = p->y_param;
    double* yTilde = p->yTilde;
    double* YTilde = p->YTilde;
    double* w = p->w;
    double theta = p->theta;
    int caching = p->caching;
    double* yTildeT = p->yTildeT;
    double* tmp_n = p->tmp_n;
    double* tmp_m = p->tmp_m;
    int m = p->m;
    int n = p->n;


    double* v_ptr = (double*) v->data;
    double* result_ptr = (double*) df->data;

    _get_weights_from_forces (w0, yTilde, v_ptr, w, caching, yTildeT, tmp_n, m, n);

    _grad_bioen_log_posterior_forces(v_ptr, w0, y_param, yTilde, YTilde, w, result_ptr, theta,
                                     caching, yTildeT, tmp_n, tmp_m, m, n);

    return;
}


void fdf_forces(const gsl_vector* x, void* params, double* f, gsl_vector* df) {
    params_t* p = (params_t*)params;

    double* w0 = p->w0;
    double* y_param = p->y_param;
    double* yTilde = p->yTilde;
    double* YTilde = p->YTilde;
    double* w = p->w;
    double theta = p->theta;
    int caching = p->caching;
    double* yTildeT = p->yTildeT;
    double* tmp_n = p->tmp_n;
    double* tmp_m = p->tmp_m;
    int m = p->m;
    int n = p->n;

    double* v_ptr = (double*) x->data;
    double* result_ptr = (double*) df->data;

    // 1) compute weights
    _get_weights_from_forces (w0, yTilde, v_ptr, w, caching, yTildeT, tmp_n, m, n);

    // 2) compute function
    *f = _bioen_log_posterior_forces(v_ptr, w0, y_param, yTilde, YTilde, w, NULL,
                                             theta, caching, yTildeT, tmp_n, tmp_m, m, n);
    // 3) compute function gradient
    _grad_bioen_log_posterior_forces(v_ptr, w0, y_param, yTilde, YTilde, w, result_ptr, theta,
                                     caching, yTildeT, tmp_n, tmp_m, m, n);

}
#endif


double _opt_bfgs_forces(double* forces, double* w0, double* y_param, double* yTilde,
                        double* YTilde, double* result, double theta, int m, int n,
                        struct gsl_config_params config, struct caching_params caching,
                        struct visual_params visual) {
    double final_val = 0.;

#ifdef ENABLE_GSL
    int iter;
    int status1 = 0;
    int status2 = 0;
    params_t* params = NULL;
    int status = 0;

    // Set up arguments in param_t structure
    status += posix_memalign((void**)&params, ALIGN_CACHE, sizeof(params_t));
    params->forces = forces;
    params->w0 = w0;
    params->y_param = y_param;
    params->YTilde = YTilde;
    params->yTilde = yTilde;
    params->result = result;
    params->theta = theta;
    params->yTildeT = caching.yTildeT;
    params->caching = caching.lcaching;
    params->tmp_n = caching.tmp_n;
    params->tmp_m = caching.tmp_m;
    params->m = m;
    params->n = n;

    double *w;
    status += posix_memalign((void**)&w, ALIGN_CACHE, sizeof(double) * n);
    if (w == NULL) {
        printf("ERROR; allocating w\n");
        exit(-1);
    }
    params->w = w;



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
    x0 = gsl_vector_alloc(m);

    for (size_t i = 0; i < m; i++) {
        gsl_vector_set(x0, i, forces[i]);
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

    s = gsl_multimin_fdfminimizer_alloc(T, m);
    my_func.f = _bioen_log_posterior_forces_interface;
    my_func.df = _grad_bioen_log_posterior_forces_interface;
    my_func.fdf = fdf_forces;
    my_func.n = m;
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
            // gsl_error("At fdfminimizer_iterate",__FILE__,__LINE__,status1 );
            break;
        }

        // status2 = gsl_multimin_test_gradient(s->gradient, _g_tol);
        status2 = gsl_multimin_test_gradient__scipy_optimize_vecnorm(s->gradient, config.tol);

        iter++;

    } while (status2 == GSL_CONTINUE && iter < config.max_iterations);

    // Get the final minimizing function parameters
    const gsl_vector* x = NULL;
    x = gsl_multimin_fdfminimizer_x(s);

    // Get minimum value
    final_val = gsl_multimin_fdfminimizer_minimum(s);

    // Copy back the result.
    for (size_t i = 0; i < m; i++) {
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

    gsl_vector_free(x0);
    gsl_multimin_fdfminimizer_free(s);
    free(params);

    free(w);

#else
    printf("GSL has not been configured properly\n");
#endif

    return final_val;
}


// LibLBFGS optimization interface
double _opt_lbfgs_forces(double* forces, double* w0, double* y_param, double* yTilde,
                         double* YTilde, double* result, double theta, int m, int n,
                         struct lbfgs_config_params config, struct caching_params caching,
                         struct visual_params visual) {
    double final_result = 0;

#ifdef ENABLE_LBFGS
    int status = 0;
    params_t* params = NULL;
    // Set up arguments in param_t structure
    status += posix_memalign((void**)&params, ALIGN_CACHE, sizeof(params_t));
    params->forces = forces;
    params->w0 = w0;
    params->y_param = y_param;
    params->YTilde = YTilde;
    params->yTilde = yTilde;
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
    int i;

    lbfgsfloatval_t fx = 0;
    lbfgsfloatval_t* x = lbfgs_malloc(m);
    lbfgs_parameter_t param;

    if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return 1;
    }

    // Initialize the variables.
    for (i = 0; i < m; i++) {
        x[i] = forces[i];
    }

    // Initialize the parameters for the L-BFGS optimization.
    lbfgs_parameter_init(&param);

    param.linesearch = config.linesearch;
    param.max_iterations = config.max_iterations;
    param.delta = config.delta;      // default: 0?
    param.epsilon = config.epsilon;  // default: 1e5
    param.ftol = config.ftol;        // default: 1e-4
    param.gtol = config.gtol;
    param.past = config.past;
    param.max_linesearch = config.max_linesearch;  // default: 20

    lbfgs_verbose_forces = visual.verbose;
    if (visual.verbose) {
        printf("\t=========================\n");
        printf("\tcaching_yTilde_tranposed : %s\n", caching.lcaching ? "enabled" : "disabled");
        printf("\tlinesearch               : %d\n", param.linesearch);
        printf("\tmax_iterations           : %d\n", param.max_iterations);
        printf("\tdelta                    : %lf\n", param.delta);
        printf("\tepsilon                  : %lf\n", param.epsilon);
        printf("\tftol                     : %lf\n", param.ftol);
        printf("\tgtol                     : %lf\n", param.gtol);
        printf("\tpast                     : %d\n", param.past);
        printf("\tmax_linesearch           : %d\n", param.max_linesearch);
        printf("\t=========================\n");
    }

    //    Start the L-BFGS optimization; this will invoke the callback functions
    //    evaluate() and progress() when necessary.

    iterations_lbfgs_forces = 0;

    start = get_wtime();
    int return_value =
        lbfgs(m, x, &fx, interface_lbfgs_forces, progress_forces, params, &param);
    end = get_wtime();

    if (visual.verbose) {
        printf("\t%s\n", lbfgs_strerror(return_value));
        printf("\tConfig: m=%d and n=%d\n", m, n);
        printf("\tCurrent function value  = %.6lf\n", fx);
        printf("\tIterations              : %zu\n", iterations_lbfgs_forces);
        printf("\tTime(s) of L-BFGS       : %.12lf\n", end - start);
        printf("\tTime(s) per iter        : %.12lf\n", (end - start) / iterations_lbfgs_forces);
    }

    final_result = fx;

    for (i = 0; i < m; i++) {
        result[i] = x[i];
    }

    lbfgs_free(x);
    free(params);

#else
    printf("LibLBFGS has not been configured properly\n");
#endif

    return final_result;
}
