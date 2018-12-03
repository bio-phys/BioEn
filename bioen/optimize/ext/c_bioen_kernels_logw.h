#ifndef KERNELS_LOGW_H
#define KERNELS_LOGW_H

/** C implementations of the log-weights method. */

#include "c_bioen_common.h"

double _get_weights(const double* const g,
                    double* const w,
                    const size_t n);

double _bioen_log_posterior_logw(const double* const g,
                                 const double* const G,
                                 const double* const yTilde,
                                 const double* const YTilde,
                                 const double* const w,
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
                                    double* const gradient,
                                    const double theta,
                                    const int caching,
                                    const double* const yTildeT,
                                    double* const tmp_n,
                                    double* const tmp_m,
                                    const int m_int,
                                    const int n_int,
                                    const double weights_sum);

double _opt_bfgs_logw(double* g,
                      double* G,
                      double* yTilde,
                      double* YTilde,
                      double* w,
                      double* result,
                      double theta,
                      int m,
                      int n,
                      struct gsl_config_params config,
                      struct caching_params caching,
                      struct visual_params visual);


double _opt_lbfgs_logw(double* g,
                    double* G,
                    double* yTilde,
                    double* YTilde,
                    double* w,
                    double* result,
                    double theta,
                    int m,
                    int n,
                    struct lbfgs_config_params config,
                    struct caching_params caching,
                    struct visual_params visual);

#endif
