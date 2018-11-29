
#include <string.h>
#include "c_bioen_error.h"
#ifdef ENABLE_GSL
#include <gsl/gsl_errno.h>
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
        // Management of GSL errors
        case GSL:
            if (value != GSL_SUCCESS) {
                printf ("Section GSL; %s\n", gsl_strerror(value));
                longjmp(*ctx,value);
            }
            break;
        // Management of LibLBFGS  errors
        //case LLBFGS:
        //    printf ("Section LLBFGS, Value %d\n", value);
        //    break;
        default:
            break;
    }

}
