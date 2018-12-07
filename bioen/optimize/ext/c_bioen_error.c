
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

//jmp_buf ctx;
jmp_buf *ctx;

void _set_ctx(jmp_buf* param_ctx) {
    ctx = param_ctx;
}


void bioen_manage_error(int section, int value) {

    switch (section) {
        // Management of POSIX errors
        case POSIX:
            if (value != 0 ) {
                printf ("POSIX_mem_alloc,  Error Value %d\n", value);
                longjmp(*ctx,value);
            }
            break;
        #ifdef ENABLE_GSL
        // Management of GSL errors
        case GSL:
            if (value != GSL_SUCCESS) {
                //printf ("Warning code!  %s\n", gsl_strerror(value));
                // Should break execution?
                //longjmp(*ctx,value);
            }
            break;
        #endif
        // Management of LibLBFGS  errors
        #ifdef ENABLE_LBFGS
        case LLBFGS:
            if (value != LBFGS_CONVERGENCE ) {
                lbfgs_strerror(value);
                // Should break execution?
                //longjmp(*ctx,value);
            }
        #endif
        default:
            break;
    }

}

#ifdef ENABLE_GSL
// Error handler for gsl's BFGS algorithm.
void handler(const char* reason, const char* file, int line, int gsl_errno) {
    printf("GSL; error has occured\n");
    printf("----  %s\n", reason);
    printf("----  ErrNo   : %d\n", gsl_errno);
    printf("----  ErrDesc :\"%s\"\n", gsl_strerror(gsl_errno));
    longjmp(*ctx,gsl_errno);
    return;
}
// Implementation of the norm as it is used by the SciPy minimizer,
// see vecnorm() in optimize.py.
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
            //        case LBFGS_SUCCESS                     :  return
            //        "LBFGS_SUCCESS";  // 0
        case LBFGS_CONVERGENCE:
            // return "Success: reached convergence (gtol)";    // 0
            return "LBFGS_CONVERGENCE";  // 0
        case LBFGS_STOP:
            // return "Success: met stopping criteria (ftol).";
            return "LBFGS_STOP";
        case LBFGS_ALREADY_MINIMIZED:
            return "The initial variables already minimize the objective "
                   "function.";
        case LBFGSERR_UNKNOWNERROR:
            return "Unknown error.";
        case LBFGSERR_LOGICERROR:
            return "Logic error.";
        case LBFGSERR_OUTOFMEMORY:
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
        case LBFGSERR_INCREASEGRADIENT:
            return "The current search direction increases the objective "
                   "function value.";
        default:
            return "(unknown)";
    };
}
#endif
