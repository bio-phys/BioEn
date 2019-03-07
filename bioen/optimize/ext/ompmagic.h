#ifndef OMPMAGIC_H
#define OMPMAGIC_H

// Preprocessor macros to toggle OpenMP parallel and SIMD regions at compile
// time, independent of the compiler switches used outside.
//
// Problem: One cannot simply use 'pragma omp ...' as a preprocessor macro
// --- which would make it easy to implement a toggling of OpenMP --- because of
// nifty substitution orders and rules.
//
// Solution: The macros below use _Pragma() which is C99 compliant in
// combination with stringification to implement wrapper macros for the most
// required OpenMP building blocks.  Simply use them as macro functions, passing
// OpenMP clauses as arguments.
//
// Useful to perform debugging of parallel and SIMD reductions, for example.
//
// Main repository: https://gitlab.mpcdf.mpg.de/khr/ompmagic

#define OMP_SCHEDULE schedule(static)

// handle traditional OpenMP thread parallel regions and loops
#if USE_OMP_THREADS == 1
#define OMP_P omp
#define VARS_FOR(...) __VA_ARGS__
#else
#define OMP_P off
#define VARS_FOR(...)
#endif
// handle OpenMP simd loops
#if USE_OMP_VECTORS == 1
#define OMP_S omp
#define VARS_SIMD(...) __VA_ARGS__
#else
#define OMP_S off
#define VARS_SIMD(...)
#endif
// handle mixed cases of OpenMP threading and simd
#if USE_OMP_THREADS == 1 && USE_OMP_VECTORS == 1
#define OMP_Q omp
#define FOR_SIMD for \
    simd
#else
#if USE_OMP_THREADS == 1
#define OMP_Q omp
#define FOR_SIMD for
#elif USE_OMP_VECTORS == 1
#define OMP_Q omp
#define FOR_SIMD simd
#else
#define OMP_Q off
#endif
#endif

#define JOIN(X, Y) X Y
#define XSTR(X) #X
#define STR(X) XSTR(X)

#define PRAGMA_OMP_PARALLEL(...) _Pragma(STR(JOIN(OMP_P parallel, __VA_ARGS__)))

#define PRAGMA_OMP_FOR(...) \
   _Pragma( STR( JOIN(OMP_P for, __VA_ARGS__) ) )

#define PRAGMA_OMP_SIMD(...) _Pragma(STR(JOIN(OMP_S simd, __VA_ARGS__)))

#define PRAGMA_OMP_FOR_SIMD(...) _Pragma(STR(JOIN(OMP_Q FOR_SIMD, __VA_ARGS__)))

#define PRAGMA_OMP_ORDERED(...) _Pragma(STR(JOIN(OMP_P ordered, __VA_ARGS__)))

#define PRAGMA_OMP_ATOMIC(...) _Pragma(STR(JOIN(OMP_P atomic, __VA_ARGS__)))

#define PRAGMA_OMP_CRITICAL(...) _Pragma(STR(JOIN(OMP_P critical, __VA_ARGS__)))

#define PRAGMA_OMP_SINGLE(...) _Pragma(STR(JOIN(OMP_P single, __VA_ARGS__)))

#endif
