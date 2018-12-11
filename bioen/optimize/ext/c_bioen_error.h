
#include <stdio.h>
#include "c_bioen_common.h"

#ifdef ENABLE_GSL
const char* bioen_gsl_error(int);
#endif

#ifdef ENABLE_LBFGS
char *lbfgs_strerror(int);
#endif

static const char message_gsl_unavailable[] = "BioEN optimize was not compiled with GSL.";
static const char message_lbfgs_unavailable[] =
    "BioEN optimize was not compiled with liblbfgs.";


