/** C implementations of common functions for the forces and the logw methods. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#ifdef ENABLE_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#endif

#ifdef ENABLE_LBFGS
#include <lbfgs.h>
#endif

#include "c_bioen_common.h"
#include "ompmagic.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef ENABLE_GSL
// GSL bfgs algorithm (default)
int _gsl_multimin_algorithm = fdfminimizer_vector_bfgs2;
#endif

int _library_gsl() {
#ifdef ENABLE_GSL
    return 1;
#endif
    return 0;
}

int _library_lbfgs() {
#ifdef ENABLE_LBFGS
    return 1;
#endif
    return 0;
}

// flag to toggle aggressive OpenMP parallelization, default: 0 == off
int _fast_openmp_flag = 0;

void _set_fast_openmp_flag(int flag) {
    _fast_openmp_flag = flag;
};

int _get_fast_openmp_flag() {
    return _fast_openmp_flag;
};

void _omp_set_num_threads(int flag) {
#ifdef _OPENMP
    omp_set_num_threads(flag);
#endif
};

// get the time since the epoch in microseconds
double get_wtime(void) {
    struct timeval time_mark;
    gettimeofday(&time_mark, NULL);
    return (double)time_mark.tv_sec + ((double)time_mark.tv_usec) * 1.e-6;
}

double _bioen_chi_squared(const double* const w, const double* const yTilde,
                          const double* const YTilde, double* const tmp_m, const size_t m,
                          const size_t n) {
    double val = 0.0;

    if (_fast_openmp_flag) {
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR(OMP_SCHEDULE reduction(+ : val))
            for (size_t i = 0; i < m; i++) {
                double me = 0.0;
                PRAGMA_OMP_SIMD(reduction(+ : me))
                for (size_t j = 0; j < n; j++) {
                    me += yTilde[i * n + j] * w[j];
                }
                me -= YTilde[i];
                tmp_m[i] = me * me;
                val += tmp_m[i];
            }
        }
    } else {
        PRAGMA_OMP_PARALLEL(default(shared)) {
            PRAGMA_OMP_FOR(OMP_SCHEDULE)
            for (size_t i = 0; i < m; i++) {
                double me = 0.0;
                PRAGMA_OMP_SIMD(reduction(+ : me))
                for (size_t j = 0; j < n; j++) {
                    me += yTilde[i * n + j] * w[j];
                }
                me -= YTilde[i];
                tmp_m[i] = me * me;
            }
        }
        for (size_t i = 0; i < m; i++) {
            val += tmp_m[i];
        }
    }

    return (0.5 * val);
}


#ifdef ENABLE_GSL
int gsl_multimin_test_gradient__scipy_optimize_vecnorm(const gsl_vector* g, double epsabs) {
    double norm;
    double temp;
    size_t n_elem;
    double* data;
    size_t i;

    if (epsabs < 0.0) {
        GSL_ERROR("absolute tolerance is negative", GSL_EBADTOL);
    }

    n_elem = g->size;
    data = g->data;
    norm = 0.0;
    for (i = 0; i < n_elem; ++i) {
        temp = fabs(data[i]);
        if (temp > norm) {
            norm = temp;
        }
    }

    if (norm < epsabs) {
        return GSL_SUCCESS;
    } else {
        return GSL_CONTINUE;
    }
}
#endif