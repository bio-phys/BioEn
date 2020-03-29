#!/usr/bin/env python

################################################
#
#   BioEn - run_bioen.py
#
################################################

import sys
import numpy as np
import random as rdm
import time
import logging

from . import utils
from . import procedure
from .observables import observables
from .show_plot import show_plot


def get_thetas(theta):
    """
    Converts theta information to array.

    Parameters
    ----------
    theta: single float, list of floats or file (containing theta values)

    Returns
    -------
    thetas: array with theta(s)
    """
    if ',' in theta:
        return [float(t) for t in theta.split(',')]
    elif utils.is_float(theta) or theta == '0.0':
        return [float(theta)]
    elif '.dat' in theta:
        return np.loadtxt(theta, unpack=True, ndmin=1)
    else:
        raise ValueError('Please provide theta value(s) in an appropriate format.')


def main():
    try:
        main_function()
    except Exception as e:
        print("Exception occurred!")
        # print("Error Message: ")
        # print(e)
        raise
        sys.exit(1)  # indicate error via non-zero exit status


def main_function():
    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    # optimization options
    parser.add_option('--optimization_method',
                      dest='opt_method',
                      default='log-weights',
                      choices=['log-weights', 'forces'],
                      help='Choose optimization method: log-weights (default) or forces.')
    parser.add_option('--optimization_minimizer',
                      dest='opt_minimizer',
                      default='scipy',
                      choices=['scipy', 'GSL', 'gsl', 'lbfgs'],
                      help='Choose for the minimizer of the optimization: '
                      'scipy (default), GSL or LBFGS.')
    parser.add_option('--optimization_algorithm',
                      dest='opt_algorithm',
                      default='bfgs',
                      help='Choose for optimization algorithm according to minimizer: '
                      'scipy: bfgs, lbfgs, cg; '
                      'GSL: conjugate_fr, conjugate_pr, bfgs, bfgs, steepest_descent; '
                      'LBFGS: lbfgs.')
    parser.add_option('--optimization_parameters',
                      dest='opt_parameter_mod',
                      default='',
                      help='Adapt specific parameters of the minimizer to override the defaults specified in <bioen_optimize.yaml>. '
                      'Example argument: gsl:step_size=0.01;gsl:tol=0.001')
    parser.add_option('--optimization_verbose',
                      action='store_true',
                      dest='opt_verbose',
                      default=False,
                      help='Verbose output of the optimization.')
    parser.add_option('--reference_weights',
                      dest='reference_weights',
                      default='uniform',
                      help='Choose for reference weights: uniform (uniformly distributed; default), '
                      'random (randomly distributed), or a file (e.g. <path_to_file>/weights.dat).')
    parser.add_option('--initial_weights',
                      dest='initial_weights',
                      default='uniform',
                      help='Choose for initial weights: uniformly distributed (default), '
                      'randomly distributed, or a file (e.g. <path_to_file>/weights.dat).')

    parser.add_option('--output_pkl',
                      dest='output_pkl_fn',
                      default='bioen_result.pkl',
                      help='Name of output pkl file (includes weighted simulated data, '
                      'weights, theta, relative entropy, chi2, and much more information).'
                      'The default is bioen_result.pkl.')
    parser.add_option('--output_weights',
                      dest='output_weights_fn',
                      default='bioen_result_weights.dat',
                      help='Name of output weights file, which can be used as input for '
                      'a following run (e.g. <path_to_file>/weights.dat)')
    parser.add_option('--output_models_weights',
                      dest='output_models_weights_fn',
                      type='string',
                      default='bioen_result_models_weights.dat',
                      help='Name of output file, which contains model number and weights '
                      ' (e.g. out-models-weights.dat (default)).')
    parser.add_option('--output_pkl_input_data',
                      dest='output_pkl_input_fn',
                      default=None,
                      help='Name of pkl file, which can be used as input pkl in '
                      'following run (to save time in loading simulated data).')

    parser.add_option('--theta',
                      dest='theta',
                      default='10.0',
                      help='Provide information on the confidence value (theta). You can '
                      'provide a file with thetas (e.g. theta.dat), list of thetas like '
                      '30,20,10, or a single value. If no value is provided, a default '
                      'of 10.0 is used')

    # general options about experimental and simulated data
    parser.add_option('--number_of_models',
                      dest='nmodels',
                      type=int,
                      default=None,
                      help='Required: define number of models/conformers.')
    parser.add_option('--models_list',
                      dest='models_list_fn',
                      type='string',
                      default=None,
                      help='Required: provide file with list of models/conformers.')
    parser.add_option('--experiments',
                      dest='experiments',
                      default=None,
                      help='Required: provide at least one of the following experimental '
                      'methods: generic, cd, deer, scattering.')
    parser.add_option('--input_pkl',
                      dest='input_pkl_fn',
                      default=None,
                      help='Name of input pkl file (e.g. <path_to_file>/input.pkl)')

    # generic - experimental measurements with standard input
    parser.add_option('--sim_path',
                      dest='generic_sim_path',
                      default=None,
                      help='Path of files with simulated data (in a generic format).')
    parser.add_option('--sim_prefix',
                      dest='generic_sim_prefix',
                      type='string',
                      default='sim',
                      help='Prefix of files with simulated data (e.g. sim (default).')
    parser.add_option('--sim_suffix',
                      dest='generic_sim_suffix',
                      type='string',
                      default='generic',
                      help='Suffix of files with simulated data (e.g. generic (default).')
    parser.add_option('--exp_path',
                      dest='generic_exp_path',
                      default=None,
                      help='Path of files with experimental data for generic data input.')
    parser.add_option('--exp_prefix',
                      dest='generic_exp_prefix',
                      type='string',
                      default='exp',
                      help='Prefix of files with experimental data for generic data input. '
                      ' (e.g. exp (default).')
    parser.add_option('--exp_suffix',
                      dest='generic_exp_suffix',
                      type='string',
                      default='generic',
                      help='Suffix of files with experimental data for generic data input. '
                      '(e.g. generic (default).')
    parser.add_option('--data_input_pkl',
                      dest='generic_in_pkl',
                      type='string',
                      default=None,
                      help='Input pkl file with experimental and simulated generic data.')
    parser.add_option('--data_output_pkl',
                      dest='generic_out_pkl',
                      type='string',
                      default=None,
                      help='Output pkl file with experimental and simulated generic data.')
    parser.add_option('--data_ids',
                      dest='generic_data_ids',
                      default=None,
                      help='Provide either a list of data ids e.g., "noe_1,noe_2,distance_1" or '
                      ' define \"all\" (all experimental data in your exp data file is used).')
    parser.add_option('--data_weight',
                      dest='generic_data_weight',
                      type=float,
                      default=1.0,
                      help='Weight of the generic data (in case of ensemble refinement '
                      'weighting with different experimental data).')
    
    # circular dichroism
    parser.add_option('--cd_sim_path',
                      dest='cd_sim_path',
                      default=None,
                      help='Path of files with simulated data CD data.')
    parser.add_option('--cd_sim_prefix',
                      dest='cd_sim_prefix',
                      type='string',
                      default='sim',
                      help='Prefix of files with simulated CD data (e.g. sim (default).')
    parser.add_option('--cd_sim_suffix',
                      dest='cd_sim_suffix',
                      type='string',
                      default='cd',
                      help='Suffix of files with simulated CD data (e.g. cd (default).')
    parser.add_option('--cd_exp_path',
                      dest='cd_exp_path',
                      default=None,
                      help='Path of files with experimental data for CD data input.')
    parser.add_option('--cd_exp_prefix',
                      dest='cd_exp_prefix',
                      type='string',
                      default='exp',
                      help='Prefix of files with experimental data for CD data input. '
                      ' (e.g. exp (default).')
    parser.add_option('--cd_exp_suffix',
                      dest='cd_exp_suffix',
                      type='string',
                      default='cd',
                      help='Suffix of files with experimental data for CD data input. '
                      '(e.g. cd (default).')
    parser.add_option('--cd_noise',
                      dest='cd_noise',
                      type='string',
                      default='0.01',
                      help='Define noise level (sigma) of the DEER data. '
                      'Define either single value for all data points (e.g. 0.01 (default)), '
                      'the difference between experimental measurement and fit (\"exp_fit_dif\"), '
                      'the standard deviation of the difference between experimental measurement , '
                      'and fit (\"exp_fit_std\"), or a file with sigmas for each data point '
                      '(e.g. <path_to_file>/err.dat).')
    parser.add_option('--cd_data_input_pkl',
                      dest='cd_in_pkl',
                      type='string',
                      default=None,
                      help='Input pkl file with experimental and simulated CD data.')
    parser.add_option('--cd_data_output_pkl',
                      dest='cd_out_pkl',
                      type='string',
                      default=None,
                      help='Output pkl file with experimental and simulated CD data.')
    parser.add_option('--cd_data_weight',
                      dest='cd_data_weight',
                      type=float,
                      default=1.0,
                      help='Weight of the CD data (in case of ensemble refinement '
                      'with different experimental data).')

    # deer
    parser.add_option('--deer_sim_path',
                      dest='deer_sim_path',
                      default=None,
                      help='Path to simulated DEER files.')
    parser.add_option('--deer_sim_prefix',
                      dest='deer_sim_prefix',
                      type='string',
                      default='conf',
                      help='Prefix of simulated DEER files (e.g. conf (default).')
    parser.add_option('--deer_sim_suffix',
                      dest='deer_sim_suffix',
                      type='string',
                      default='deer',
                      help='Suffix of simulated DEER files (e.g. deer (default).')
    parser.add_option('--deer_exp_path',
                      dest='deer_exp_path',
                      default=None,
                      help='Path to experimental DEER files.')
    parser.add_option('--deer_exp_prefix',
                      dest='deer_exp_prefix',
                      type='string',
                      default='exp',
                      help='Prefix of experimental DEER files (e.g. exp (default).')
    parser.add_option('--deer_exp_suffix',
                      dest='deer_exp_suffix',
                      type='string',
                      default='deer',
                      help='Suffix of experimental DEER files (e.g. deer (default).')
    parser.add_option('--deer_labels',
                      dest='deer_labels',
                      default=None,
                      help='List of labeled residue pairs (e.g. 370+259,370+292,370+265).')
    parser.add_option('--deer_noise',
                      dest='deer_noise',
                      type='string',
                      default='0.01',
                      help='Define noise level (sigma) of the DEER data. '
                      'Define either single value for all data points (e.g. 0.01 (default)), '
                      'the difference between experimental measurement and fit (\"exp_fit_dif\"), '
                      'the standard deviation of the difference between experimental measurement , '
                      'and fit (\"exp_fit_std\"), or a file with sigmas for each data point '
                      '(e.g. <path_to_file>/err.dat).')
    parser.add_option('--deer_modulation_depth',
                      dest='deer_moddepth',
                      type='string',
                      default='0.15',
                      help='Define modulation depths of the DEER data. Define either single '
                      'value for all data points (e.g. 0.15 (default)) '
                      'a file with sigmas for each label (e.g. <path_to_file>/moddepth.dat), '
                      'or let BioEn perform an initial optimization for the modulation depth '
                      '(\"initial-optimization\").')
    parser.add_option('--deer_data_weight',
                      dest='deer_data_weight',
                      type=float,
                      default=1.0,
                      help='Weight of the DEER data (in case of ensemble refinement with different '
                      'experimental data).')
    parser.add_option('--deer_input_pkl',
                      dest='deer_in_pkl',
                      type='string',
                      default=None,
                      help='Input pkl file with experimental and simulated DEER data.')
    parser.add_option('--deer_input_hd5',
                      dest='deer_in_hd5',
                      type='string',
                      default=None,
                      help='Input hd5 file with experimental and simulated DEER data.')
    parser.add_option('--deer_output_pkl',
                      dest='deer_out_pkl',
                      type='string',
                      default=None,
                      help='Output pkl file with experimental and simulated DEER data.')
    parser.add_option('--deer_input_sim_pkl',
                      dest='deer_in_sim_pkl',
                      type='string',
                      default=None,
                      help='Input pkl with dictionary of simulated DEER data '
                      '(e.g. sim_tmp[nmodel][label].')
    parser.add_option('--deer_input_sim_hd5',
                      dest='deer_in_sim_hd5',
                      type='string',
                      default=None,
                      help='Input hd5 with dictionary of simulated DEER data '
                      '(e.g. sim_tmp[nmodel][label].')

    # scattering
    parser.add_option('--scattering_sim_path',
                      dest='scattering_sim_path',
                      default=None,
                      help='Path to files with simulated scattering data.')
    parser.add_option('--scattering_sim_prefix',
                      dest='scattering_sim_prefix',
                      type='string',
                      default='conf',
                      help='Prefix of simulated scattering files (e.g. conf (default).')
    parser.add_option('--scattering_sim_suffix',
                      dest='scattering_sim_suffix',
                      type='string',
                      default='scattering',
                      help='Suffix of simulated scattering files (e.g. scattering (default).')
    parser.add_option('--scattering_exp_path',
                      dest='scattering_exp_path',
                      default=None,
                      help='Path to experimental scattering files.')
    parser.add_option('--scattering_exp_prefix',
                      dest='scattering_exp_prefix',
                      type='string',
                      default='exp',
                      help='Prefix of experimental scattering files (e.g. exp (default).')
    parser.add_option('--scattering_exp_suffix',
                      dest='scattering_exp_suffix',
                      type='string',
                      default='scattering',
                      help='Suffix of experimental scattering files (e.g. scattering (default).')
    parser.add_option('--scattering_noise',
                      dest='scattering_noise',
                      type='string',
                      default='0.01',
                      help='Define noise level (sigma) of the scattering data. '
                      'Define either single value for all data points (e.g. 0.01 (default)), '
                      'provide the third column in the exp data file (\"exp-file\"), '
                      'or a file with error for each data point (e.g. <path_to_file>/err.dat).')
    parser.add_option('--scattering_scaling_factor',
                      dest='scattering_scaling_factor',
                      type='string',
                      default='0.0002',
                      help='Define initial value for the scaling factor of the scattering data. '
                      'Define either single value (e.g. 0.0002 (default)) '
                      'or let BioEn perform an initial optimization for the nuisance parameter '
                      '(\"initial-optimization\").')
    parser.add_option('--scattering_additive_constant',
                      dest='scattering_additive_constant',
                      type=float,
                      default='0.0002',
                      help='If needed, define initial value for additive constant for the '
                      'scattering data.')
    parser.add_option('--scattering_solvent',
                      dest='scattering_solvent',
                      type=float,
                      default='0.0002',
                      help='If needed, define initial value for the solvent for the '
                      'scattering data.')
    parser.add_option('--scattering_data_weight',
                      dest='scattering_data_weight',
                      type=float,
                      default=1.0,
                      help='Weight of the scattering data (in case of ensemble refinement '
                      'with different experimental data).')
    parser.add_option('--scattering_input_pkl',
                      dest='scattering_in_pkl',
                      type='string',
                      default=None,
                      help='Input pkl file with experimental and simulated scattering data.')
    parser.add_option('--scattering_input_hd5',
                      dest='scattering_in_hd5',
                      type='string',
                      default=None,
                      help='Input hd5 file with experimental and simulated scattering data.')
    parser.add_option('--scattering_output_pkl',
                      dest='scattering_out_pkl',
                      type='string',
                      default=None,
                      help='Output pkl file with experimental and simulated scattering data.')
    parser.add_option('--scattering_input_sim_pkl',
                      dest='scattering_in_sim_pkl',
                      type='string',
                      default=None,
                      help='Input pkl with dictionary of simulated scattering data '
                      '(e.g. sim_tmp[nmodel][label].')
    parser.add_option('--scattering_input_sim_hd5',
                      dest='scattering_in_sim_hd5',
                      type='string',
                      default=None,
                      help='Input hd5 with dictionary of simulated scattering data '
                      '(e.g. sim_tmp[nmodel][label].')
    
    parser.add_option('--number_of_iterations',
                      dest='iterations',
                      type=int,
                      default=1,
                      help='Define number of iterations for reweighting. It is useful if'
                      'nuisance parameter is relevant with the exerimental data'
                      '(e.g. DEER/PELDOR or SAXS).')


    # options for a simple check of the BioEn reweighting
    parser.add_option('--simple_plot',
                      action='store_true',
                      dest='simple_plot',
                      help='Simple plot for all kind of data. Visualizes experimental data '
                      'and ensemble averaged data from BioEn optimization. If this option '
                      'is set, no BioEn reweighting is performed (only visualization).')
    parser.add_option('--simple_plot_input',
                      dest='bioen_pkl',
                      type='string',
                      default=None,
                      help='Define BioEn result file in pkl format.')
    parser.add_option('--simple_plot_output',
                      dest='simple_plot_output',
                      type='string',
                      default='test.pdf',
                      help='Define filename of simple plot output (test.pdf (default)).')

    options, args = parser.parse_args()

    if options.simple_plot is True:
        if not options.bioen_pkl or 'pkl' not in options.bioen_pkl:
            parser.error('Please provide a BioEn file in pkl format.')
        show_plot.Show_plot(options.bioen_pkl, options.simple_plot_output)
    else:
        if not options.experiments:
            parser.error('Please provide information on experiments (--experiments).')
        if not options.models_list_fn:
            parser.error('Please provide file with list of models (--modelsList).')

        options.thetas = get_thetas(options.theta)
        obs = observables.Observables(options)

        logging.basicConfig(level=logging.INFO)
        logging.info('BioEn weight refinement runs with')
        logging.info('    optimization method: {}'.format(options.opt_method))
        logging.info('    optimization algorithm: {}'.format(options.opt_algorithm))
        logging.info('    optimization minimizer: {}'.format(options.opt_minimizer))
        logging.info('    optimization parameter modification: {}'.format(options.opt_parameter_mod))
        start_time = time.time()
        procedure.start_reweighting(options, obs)
        end_time = time.time()
        logging.info("BioEn finished weight refinement in {:.2f} s.".format(end_time - start_time))


if __name__ == "__main__":
    main()
