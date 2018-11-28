#ifndef KERNELS_LOGW_H
#define KERNELS_LOGW_H

/** C implementations of the log-weights method. */

#include "c_bioen_common.h"

double _get_weights(const double* const,
                    double* const,
                    const size_t);

double _bioen_log_posterior_logw(const double* const g,
                                 const double* const G, 
                                 const double* const yTilde, 
                                 const double* const YTilde,
                                 const double* const w, 
                                 const double* const t1, 
                                 const double* const t2, 
                                 const double* const dummy,
                                 const double theta, 
                                 const int caching, 
                                 const double* const yTildeT, 
                                 double* const tmp_n,
                                 double* const tmp_m, 
                                 const int m_int, 
                                 const int n_int,
                                 const double weights_sum);

void _grad_bioen_log_posterior_logw(const double* const g,
                                    const double* const G,
                                    const double* const yTilde,
                                    const double* const YTilde,
                                    const double* const w,
                                    double* const t1,
                                    double* const t2,
                                    double* const gradient,
                                    const double theta,
                                    const int caching,
                                    const double* const yTildeT,
                                    const double* const tmp_n,
                                    const double* const tmp_m,
                                    const int m_int,
                                    const int n_int,
                                    const double weights_sum);

double _opt_bfgs_logw(double *, double *, double *, double *, double *,
                      double *, double *, double *, double, int, int,
                      struct gsl_config_params, struct caching_params,
                      struct visual_params);
double _opt_lbfgs_logw(double *, double *, double *, double *, double *,
                       double *, double *, double *, double, int, int,
                       struct lbfgs_config_params, struct caching_params,
                       struct visual_params);

#endif
