#ifndef COMMON_H
#define COMMON_H

/** C implementations of  [grad]_log_posterior[_forces/_log_weights], header
 * file.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#ifdef ENABLE_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_multimin.h> //Multidimensional minimization
#endif

#ifdef ENABLE_LBFGS
#include <lbfgs.h>
#endif

#define ALIGN_CACHE 64

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
    "fdfminimizer_conjugate_fr", "fdfminimizer_conjugate_pr",

    "fdfminimizer_vector_bfgs2", "fdfminimizer_vector_bfgs",

    "fdfminimizer_steepest_descent"};

int gsl_multimin_test_gradient__scipy_optimize_vecnorm(const gsl_vector *,
                                                       double);
#endif

typedef struct params_t {
  double *forces;
  double *w0;
  double *y_param;
  double *g;
  double *G;
  double *yTilde;
  double *YTilde;
  double *w;
  double *t1;
  double *t2;
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
  size_t max_iterations;
  size_t algorithm;
} gsl_config_params;

typedef struct lbfgs_config_params {
  size_t linesearch;
  size_t max_iterations;
  double delta;
  double epsilon;
  double ftol;
  double gtol;
  int past;
  int max_linesearch;
} lbfgs_config_params;

typedef struct caching_params {
  size_t lcaching;
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

// Shared functions
double get_wtime(void);
double _bioen_chi_squared(double *, double *, double *, double *, size_t,
                          size_t);

#ifdef ENABLE_GSL
void handler(const char *, const char *, int, int);
#endif

#ifdef ENABLE_LBFGS
char *lbfgs_strerror(int);
#endif

#endif
