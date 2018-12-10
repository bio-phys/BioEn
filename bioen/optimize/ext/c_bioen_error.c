
#include <string.h>
#include "c_bioen_error.h"
#ifdef ENABLE_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#endif

#ifdef ENABLE_LBFGS
#include <lbfgs.h>
#endif

char* lbfgs_strerror(int error);

const char* bioen_gsl_error(int gsl_errno){
    return gsl_strerror(gsl_errno);
}


#ifdef ENABLE_GSL

// Error enum from -2 to 32
// Continue/Failure/Success = -2 to 0
// 1 to 32 : error codes

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
    }

    return GSL_CONTINUE;
}
#endif

#ifdef ENABLE_LBFGS
// Error values description for the L-BFGS algorithm
char* lbfgs_strerror(int error) {

    switch (error) {
            // case LBFGS_SUCCESS: // 0
        case LBFGS_CONVERGENCE:                                 // 0
            // return "Success: reached convergence (gtol)";
            return "LBFGS_CONVERGENCE";
        case LBFGS_STOP:                                        // 1
            // return "Success: met stopping criteria (ftol).";
            return "LBFGS_STOP";
        case LBFGS_ALREADY_MINIMIZED:                           // 2
            return "The initial variables already minimize the objective "
                   "function.";
        case LBFGSERR_UNKNOWNERROR:                             // -1024
            return "Unknown error.";
        case LBFGSERR_LOGICERROR:                               // -1023
            return "Logic error.";
        case LBFGSERR_OUTOFMEMORY:                              // -1022
            return "Insufficient memory.";
        case LBFGSERR_CANCELED:
            return "The minimization process has been canceled.";
        case LBFGSERR_INVALID_N:
            return "Invalid number of variables specified.";
        case LBFGSERR_INVALID_N_SSE:
            return "Invalid number of variables (for SSE) specified.";
        case LBFGSERR_INVALID_X_SSE:
            return "The array x must be aligned to 16 (for SSE).";
        case LBFGSERR_INVALID_EPSILON:
            return "Invalid parameter lbfgs_parameter_t::epsilon specified.";
        case LBFGSERR_INVALID_TESTPERIOD:
            return "Invalid parameter lbfgs_parameter_t::past specified.";
        case LBFGSERR_INVALID_DELTA:
            return "Invalid parameter lbfgs_parameter_t::delta specified.";
        case LBFGSERR_INVALID_LINESEARCH:
            return "Invalid parameter lbfgs_parameter_t::linesearch specified.";
        case LBFGSERR_INVALID_MINSTEP:
            return "Invalid parameter lbfgs_parameter_t::max_step specified";
        case LBFGSERR_INVALID_MAXSTEP:
            return "Invalid parameter lbfgs_parameter_t::max_step specified.";
        case LBFGSERR_INVALID_FTOL:
            return "Invalid parameter lbfgs_parameter_t::ftol specified.";
        case LBFGSERR_INVALID_WOLFE:
            return "Invalid parameter lbfgs_parameter_t::wolfe specified.";
        case LBFGSERR_INVALID_GTOL:
            return "Invalid parameter lbfgs_parameter_t::gtol specified.";
        case LBFGSERR_INVALID_XTOL:
            return "Invalid parameter lbfgs_parameter_t::xtol specified.";
        case LBFGSERR_INVALID_MAXLINESEARCH:
            return "Invalid parameter lbfgs_parameter_t::max_linesearch "
                   "specified.";
        case LBFGSERR_INVALID_ORTHANTWISE:
            return "Invalid parameter lbfgs_parameter_t::orthantwise_c "
                   "specified.";
        case LBFGSERR_INVALID_ORTHANTWISE_START:
            return "Invalid parameter lbfgs_parameter_t::orthantwise_start "
                   "specified.";
        case LBFGSERR_INVALID_ORTHANTWISE_END:
            return "Invalid parameter lbfgs_parameter_t::orthantwise_end "
                   "specified.";
        case LBFGSERR_OUTOFINTERVAL:
            return "The line-search step went out of the interval of "
                   "uncertainty.";
        case LBFGSERR_INCORRECT_TMINMAX:
            return "A logic error occurred; alternatively, the interval of "
                   "uncertainty";
        case LBFGSERR_ROUNDING_ERROR:
            return "A rounding error occurred; alternatively, no line-search "
                   "step"
                   " satisfies the sufficient decrease and curvature "
                   "conditions.";
        case LBFGSERR_MINIMUMSTEP:
            return "The line-search step became smaller than "
                   "lbfgs_parameter_t::min_step.";
        case LBFGSERR_MAXIMUMSTEP:
            return "The line-search step became larger than "
                   "lbfgs_parameter_t::max_step.";
        case LBFGSERR_MAXIMUMLINESEARCH:
            return "The line-search routine reaches the maximum number of "
                   "evaluations.";
        case LBFGSERR_MAXIMUMITERATION:
            return "The algorithm routine reaches the maximum number of "
                   "iterations.";
        case LBFGSERR_WIDTHTOOSMALL:
            return "Relative width of the interval of uncertainty is at most "
                   "lbfgs_parameter_t::xtol.";
        case LBFGSERR_INVALIDPARAMETERS:
            return "A logic error (negative line-search step) occurred.";
        case LBFGSERR_INCREASEGRADIENT:                             // -994
            return "The current search direction increases the objective "
                   "function value.";
        default:
            return "(unknown)";
    };
}
#endif
