#ifndef KERNELS_FORCES_H
#define KERNELS_FORCES_H

/** C implementations of [grad_]_log_posterior[_forces|_log_weights], header
 * file.
 */

#include "c_bioen_common.h"

double _bioen_log_posterior_forces(double *, double *, double *, double *,
                                   double *, double *, double*, double, int, double *,
                                   double *, double *, int, int);
void _grad_bioen_log_posterior_forces(double *, double *, double *, double *,
                                      double *, double *, double *, double, int, double *,
                                      double *, double *, int, int);

double _opt_bfgs_forces(double *, double *, double *, double *, double *,
                        double *, double, int, int, struct gsl_config_params,
                        struct caching_params, struct visual_params);
double _opt_lbfgs_forces(double *, double *, double *, double *, double *,
                         double *, double, int, int, struct lbfgs_config_params,
                         struct caching_params, struct visual_params);

void _get_weights_from_forces(double*, double*, double*, double*,
                              int, double*, double*, size_t, size_t);
#endif
