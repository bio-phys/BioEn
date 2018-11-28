#ifndef KERNELS_FORCES_H
#define KERNELS_FORCES_H

/** C implementations of [grad_]_log_posterior[_forces|_log_weights], header
 * file.
 */

#include "c_bioen_common.h"

void _get_weights_from_forces(const double* const, const double* const, const double* const,
                              double* const, const int, const double* const, double* const,
                              const size_t, const size_t);

double _bioen_log_posterior_forces(const double* const forces, const double* const w0,
                                   const double* const y_param, const double* const yTilde,
                                   const double* const YTilde, const double* const w,
                                   const double* const result, const double theta,
                                   const int caching, const double* const yTildeT,
                                   double* const tmp_n, double* const tmp_m, const int m_int,
                                   const int n_int);

void _grad_bioen_log_posterior_forces(const double* const forces, const double* const w0,
                                      const double* const y_param, const double* const yTilde,
                                      const double* const YTilde, const double* const w,
                                      double* const gradient, const double theta,
                                      const int caching, const double* const yTildeT,
                                      double* const tmp_n, double* const tmp_m, const int m_int,
                                      const int n_int);

double _opt_bfgs_forces(double*, double*, double*, double*, double*, double*, double, int, int,
                        struct gsl_config_params, struct caching_params, struct visual_params);

double _opt_lbfgs_forces(double*, double*, double*, double*, double*, double*, double, int, int,
                         struct lbfgs_config_params, struct caching_params,
                         struct visual_params);
#endif
