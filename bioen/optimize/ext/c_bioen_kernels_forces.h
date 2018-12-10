#ifndef KERNELS_FORCES_H
#define KERNELS_FORCES_H

/** C implementations of [grad_]_log_posterior[_forces|_log_weights], header
 * file.
 */

#include "c_bioen_common.h"

void _get_weights_from_forces(const double* const w0,
                              const double* const yTilde,
                              const double* const forces,
                              double* const w,
                              const int caching,
                              const double* const yTildeT,
                              double* const tmp_n,
                              const size_t m,
                              const size_t n);

double _bioen_log_posterior_forces(const double* const w0,
                                   const double* const yTilde,
                                   const double* const YTilde,
                                   const double* const w,
                                   const double* const result,
                                   const double theta,
                                   const int caching,
                                   const double* const yTildeT,
                                   double* const tmp_n,
                                   double* const tmp_m,
                                   const int m_int,
                                   const int n_int);

void _grad_bioen_log_posterior_forces(const double* const w0,
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
                                      const int n_int);


double _opt_bfgs_forces(params_t,
                        gsl_config_params,
                        visual_params,
                        int*);


double _opt_lbfgs_forces(params_t,
                         lbfgs_config_params,
                         visual_params,
                         int*);



#endif
