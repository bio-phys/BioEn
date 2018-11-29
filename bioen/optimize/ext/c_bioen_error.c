#include "c_bioen_error.h"
#ifdef ENABLE_GSL
#include <gsl/gsl_errno.h>
#endif


jmp_buf *ctx = NULL;

void _set_ctx(jmp_buf pctx) {
    ctx = pctx;
}


void bioen_manage_error(int section, int value) {

    if (ctx==NULL) return;

    switch (section) {
        // Management of POSIX errors
        case POSIX:
            if (value != 0 ) {
                printf ("POSIX_mem_alloc,  Error Value %d\n", value);
                longjmp(ctx,0);
            }
            break;
        // Management of GSL errors
        case GSL:
            printf ("Section GSL,    Value %d\n", value);
            break;
        // Management of LibLBFGS  errors
        case LLBFGS:
            printf ("Section LLBFGS, Value %d\n", value);
            break;
        default:
            break;
    }

}
