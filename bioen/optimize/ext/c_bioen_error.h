#include <stdio.h>
#include "c_bioen_common.h"

const char * bioen_gsl_error(int);
const char * lbfgs_strerror(int);

static const char message_gsl_unavailable[] = "BioEN optimize was not compiled with GSL.";
static const char message_lbfgs_unavailable[] = "BioEN optimize was not compiled with liblbfgs.";
