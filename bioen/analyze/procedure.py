from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np
import scipy.optimize as sopt
import pandas as pd
from timeit import default_timer as time

from . import utils
from .. import optimize
from .. import fileio


def start_reweighting(options, obs):
    """Reweighting procedure

    Parameters
    ----------
    options: object, provides options (settings) from user interface
    obs: object, contains all information from the observables
    """

    params = optimize.minimize.Parameters(options.opt_minimizer, options.opt_parameter_mod)
    params['cache_ytilde_transposed'] = True
    params['use_c_functions'] = True
    params['algorithm'] = options.opt_algorithm
    params['verbose'] = options.opt_verbose

    if options.output_pkl_fn is not None:
        output_all_pkl = dict()

    # w0 reference weights for entropy calculation
    w0 = utils.get_weights(options.reference_weights, options.nmodels)
    # winit initial weights to start optimization
    winit = utils.get_weights(options.initial_weights, options.nmodels)
    wopt = winit.copy()
    wopt[wopt == 0.0] = 10**-150

    log_w0 = optimize.log_weights.getGs(w0)

    # get exp and sim data (matrices, weighted, optimized nuisance parameters)
    exp = obs.exp.copy()
    sim, sim_init = obs.update_sim_init(wopt)

    log_wopt = optimize.log_weights.getGs(winit)
    if options.opt_method == 'forces':
        forces_init = exp.copy()
        forces_init[:] = 0.
        forces_init = forces_init.T
        forces = forces_init

    # check if an iterative optimization is defined
    if any(experiment in obs.experiments for experiment in ['deer', 'scattering']) \
            and options.iterations == 1:
        print("WARNING: Please increase the number of iterations, if you are using "
              "DEER or scattering data. An optimal BioEn result is only guaranteed "
              "with an iterative procedure of optimization of the weights "
              "and the nuisance parameters.")

    # bioen optimization with theta-series
    for theta in options.thetas:
        for i in range(options.iterations):
            if options.opt_method == 'log-weights':
                start = time()
                out_min = optimize.log_weights.find_optimum(log_wopt, log_w0,
                                                            sim_init, sim, exp, theta, params)
                end = time()
                wopt = out_min[0]
            elif options.opt_method == 'forces':
                wopt = winit.copy()
                start = time()
                out_min = optimize.forces.find_optimum(forces, wopt,
                                                       sim_init, sim, exp, theta, params)
                end = time()
                wopt = np.matrix(out_min[0])
                forces = np.matrix(out_min[2]).T

            if options.iterations > 1 or len(options.thetas) > 1:
                wopt_md = wopt.copy()
                wopt_md[wopt_md == 0.0] = 1e-150
                if any(experiment in obs.experiments for experiment in ['deer', 'scattering']):
                    sim, sim_init = obs.update_sim(wopt_md)

        if options.output_pkl_fn is not None:
            d = dict()
            d["optimization_method"] = options.opt_method
            d["optimization_algorithm"] = options.opt_algorithm
            d["optimization_minimizer"] = options.opt_minimizer
            d["w0"] = w0
            d["winit"] = winit
            d["wopt"] = wopt
            d["len_sim"] = (end - start)
            d["sim_init"] = obs.get_proc_sim_weights(winit)
            d["sim_wopt"] = obs.get_proc_sim_weights(wopt)
            d["S_init"] = utils.get_entropy(winit, wopt)
            d["chi2_init"] = optimize.common.chiSqrTerm(winit, sim, exp)
            d["S"] = utils.get_entropy(w0, wopt)
            d["chi2"] = optimize.common.chiSqrTerm(wopt, sim, exp)
            d["theta"] = theta
            d["nrestraints"] = obs.nrestraints
            d["nmodels"] = options.nmodels
            d["nmodels_list"] = obs.models_list
            exp_wopt = dict()
            exp_err_wopt = dict()
            for experiment in obs.experiments:
                if experiment == 'deer':
                    exp_wopt[experiment] = obs.observables[experiment].exp_tmp
                    exp_err_wopt[experiment] = obs.observables[experiment].exp_err_tmp
                    d["moddepth"] = obs.observables['deer'].moddepth
                    d["labels"] = obs.observables['deer'].labels
                elif experiment in ['scattering']:
                    exp_wopt[experiment] = obs.observables[experiment].exp_tmp
                    exp_err_wopt[experiment] = obs.observables[experiment].exp_err_tmp
                    d["scaling_factor"] = obs.observables['scattering'].scaling_factor
                elif experiment in ['generic']:
                    exp_wopt[experiment] = obs.observables[experiment].exp_tmp_dict
                    exp_err_wopt[experiment] = obs.observables[experiment].exp_err_tmp_dict
                elif experiment in ['cd']:
                    exp_wopt[experiment] = obs.observables[experiment].exp_tmp
                    exp_err_wopt[experiment] = obs.observables[experiment].exp_err_tmp
            d["exp"] = exp_wopt
            d["exp_err"] = exp_err_wopt
            output_all_pkl[theta] = d

    # TODO: add HDF5 output of dataframe
    if options.output_pkl_fn is not None:
        df = pd.DataFrame.from_dict(output_all_pkl)
        df.to_pickle(options.output_pkl_fn)

    # TODO: enable HDF5 output (trivial)
    if options.output_pkl_input_fn is not None:
        if options.opt_method == 'log_weights':
            fileio.dump(options.output_pkl_input_fn,
                        [log_wopt, log_w0, sim_init, sim, exp, w0, theta])
        elif options.opt_method == 'forces':
            fileio.dump(options.output_pkl_input_fn,
                        [forces_init, w0, sim_init, sim, exp, w0, theta])

    if options.output_weights_fn is not None:
        w_out = open(options.output_weights_fn, 'w')
        w_out.writelines("%f\n" % w for w in wopt)
        w_out.close()

    if options.output_models_weights_fn is not None:
        w_out = open(options.output_models_weights_fn, 'w')
        for i, w in enumerate(np.asarray(wopt)):
            w_out.writelines("{} {}\n".format(int(obs.models_list[i]), w[0]))

    return
