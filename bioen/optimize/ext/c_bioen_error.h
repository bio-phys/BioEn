
#include <stdio.h>
#include <setjmp.h>


enum sections { POSIX , GSL , LLBFGS };

void _set_ctx(jmp_buf*);
void bioen_manage_error(int, int);



