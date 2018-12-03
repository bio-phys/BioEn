
#include <stdio.h>
#include <setjmp.h>


enum sections { POSIX , GSL , LLBFGS };

void _set_ctx(jmp_buf*);
void bioen_manage_error(int, int);


#ifdef ENABLE_GSL
void handler(const char *, const char *, int, int);
#endif

#ifdef ENABLE_LBFGS
char *lbfgs_strerror(int);
#endif

static const char message_gsl_unavailable[] = "BioEN optimize was not compiled with GSL.";
static const char message_lbfgs_unavailable[] =
    "BioEN optimize was not compiled with liblbfgs.";


