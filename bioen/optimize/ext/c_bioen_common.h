#ifndef COMMON_H
#define COMMON_H

#ifdef ENABLE_GSL
#include <gsl/gsl_vector.h>
#endif

#define ALIGN_CACHE 64

#ifdef VERBOSE_DEBUG
#define DEBUG_PRINT(STR) \
    printf("%s():%d %s\n", __func__, __LINE__, STR);
#else
#define DEBUG_PRINT(STR)
#endif

#ifdef VERBOSE_DEBUG
#define DEBUG_CHECKPOINT() \
    printf("%s():%d\n", __func__, __LINE__);
#else
#define DEBUG_CHECKPOINT()
#endif


#ifdef ENABLE_GSL
extern int _gsl_multimin_algorithm;

enum gsl_multimin_algorithm_kinds {
    fdfminimizer_conjugate_fr = 0,
    fdfminimizer_conjugate_pr = 1,
    fdfminimizer_vector_bfgs2 = 2,
    fdfminimizer_vector_bfgs = 3,
    fdfminimizer_steepest_descent = 4
};

static const char *const gsl_multimin_algorithm_names[5] = {
    "fdfminimizer_conjugate_fr", "fdfminimizer_conjugate_pr", "fdfminimizer_vector_bfgs2",
    "fdfminimizer_vector_bfgs", "fdfminimizer_steepest_descent"};

int gsl_multimin_test_gradient__scipy_optimize_vecnorm(const gsl_vector *, double);
#endif


typedef struct params_t {
    double *forces;
    double *w0;
    double *g;
    double *G;
    double *yTilde;
    double *YTilde;
    double *w;
    double *result;
    double theta;
    double *yTildeT;
    int caching;
    double *tmp_n;
    double *tmp_m;
    int m;
    int n;
} params_t;

typedef struct gsl_config_params {
    double step_size;
    double tol;
    int max_iterations;
    int algorithm;
} gsl_config_params;

typedef struct lbfgs_config_params {
    int    linesearch;
    int    max_iterations;
    double delta;
    double epsilon;
    double ftol;
    double gtol;
    double wolfe;
    int past;
    int max_linesearch;
} lbfgs_config_params;

typedef struct caching_params {
    int    lcaching;
    double *yTildeT;
    double *tmp_n;
    double *tmp_m;
} caching_params;


typedef struct visual_params {
    size_t debug;
    size_t verbose;
} visual_params;

void _set_bfgs_gsl_algorithm(int);
int _get_bfgs_gsl_algorithm(void);

int _library_gsl(void);
int _library_lbfgs(void);

extern int _fast_openmp_flag;
void _set_fast_openmp_flag(int);
int _get_fast_openmp_flag(void);

void _omp_set_num_threads(int);

double get_wtime(void);

double _bioen_chi_squared(const double *const, const double *const, const double *const,
                          double *const, const size_t, const size_t);

#endif
