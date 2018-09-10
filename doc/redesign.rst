:orphan:

API transition after redesing
+++++++++++++++++++++++++++++

The current version implements a major API change, therefore, the utilization of
previous scripts could be affected.


Introduction
============

The different definition of libraries and methods has been renamed in order to
create a unified context.

NOTATION:

    - library => minimizer
    - method => algorithm


Currently, bioen.optimize performs the optimization in combination with
external packages. There are working interfaces for **scipy**, **GSL** and
**LIBLBFGS** external packages/libraries. The user must decide, at compilation
time, to include one or more of them to extend the bioen.optimize functionality.

Depending on the installation options, we are able to use up to three
'**minimizers**':

    - scipy (mandatory)
    - GSL   (optional)
    - LBFGS (optional)


Each '**minimizer**' applies different '**algorithm**'s:
    - scipy
        - bfgs
        - lbfgs
        - cg
    - gsl
        - conjugate_fr
        - conjugate_pr
        - bfgs2
        - bfgs
        - steepest_descent
    - lbfsg
        - l-bfgs

For every combination of 'minimizer' and 'algorithm' there is a set of
arguments affecting the behaviour of the optimizizer. e.g.: 'tolerance',
'step_size', or 'max_iterations'

API Optimizer
=============

A new argument, containing the configuration of the optimizer, is needed
in the current version. This argument is an structure (a set) that contains the
definition of the minimizer, algorithm, algorithm specific parameters and
bioen.optimizer extended features.

Algorithm specific configuration
--------------------------------

Once the user has decided which minimizer to use, a basic structure containing a
default configuration can be obtained from the minimizer module:

    >>> config = bioen.optimize.minimize.Parameters('gsl')

This set can be printed but also shown in a more readable way:

    >>> bioen_optimzie.minimize.show_params(config)

    - minimizer        gsl
    - debug            True
    - verbose          True
    - params           {'step_size': 0.01, 'tol': 0.001, 'max_iterations': 5000}
    - algorithm        gsl_multimin_fdfminimizer_vector_bfgs2
    - use_c_functions  True
    - n_threads        -1
    - cache_ytilde_transposed  auto

The structure contains a specific field named 'params' with the specific
configuration for the algorithm.

It is possible to modify the values within this structure except for the
minimizer field. In order to modify the values, simply index the content from
the container as follows:

    >>> config['algorithm'] = 'conjugate_pr'
    >>> config['params']['step_size']=0.05
    >>> config['params']['max_iterations']=1000


Extended features
-----------------

Along with the algorithm specific configuration there are the following options:

    - debug; extra parameters are returned from the minimizer
    - verbose; extra information is printed on the execution.
    - use_c_functions; scipy exclusive parameter to select between python or C implementation
    - n_threads; explicit number or implicit -1 (through the environment variable OMP_NUM_THREADS)
    - cache_ytilde_transposed; True: performance improvement but higher memory consumption. [true,false,auto]



Usage
-----

To apply the optimization::

    import bioen.optimize as bop

    bop.forces.find_optimum (forces_init, w0, y, yTilde, YTilde, theta, config)

    bop.log_weights.find_optimum (GInit, G, y, yTilde, YTilde, theta, config)
