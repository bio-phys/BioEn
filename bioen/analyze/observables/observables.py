from __future__ import print_function
import sys
import numpy as np
from scipy.optimize import leastsq

from .generic import generic
from .cd_data import cd_data
from .deer import deer
from .scattering import scattering
from ... import optimize


class Observables:
    def __init__(self, options):
        """
        Loads all information needed about experimental data.

        Parameters
        ----------
        options: object, provided options (settings) from user interface

        Returns
        -------
        obs: object, contains all information about experimental data
        """
        self.experiments = get_experiments(options.experiments)
        self.models_list = get_models_list(options.models_list_fn, options.nmodels)

        self.observables = dict()
        
        if 'generic' in self.experiments:
            kwargs = dict()
            kwargs['sim_path']    = options.generic_sim_path
            kwargs['sim_prefix']  = options.generic_sim_prefix
            kwargs['sim_suffix']  = options.generic_sim_suffix
            kwargs['exp_path']    = options.generic_exp_path
            kwargs['exp_prefix']  = options.generic_exp_prefix
            kwargs['exp_suffix']  = options.generic_exp_suffix
            kwargs['data_ids']    = options.generic_data_ids
            kwargs['data_weight'] = options.generic_data_weight
            kwargs['in_pkl']      = options.generic_in_pkl
            kwargs['out_pkl']     = options.generic_out_pkl
            kwargs['nmodels']     = options.nmodels
            kwargs['models_list'] = self.models_list
            self.observables['generic'] = generic.Generic(kwargs)
    
        if 'cd' in self.experiments:
            kwargs = dict()
            kwargs['sim_path']    = options.cd_sim_path
            kwargs['sim_prefix']  = options.cd_sim_prefix
            kwargs['sim_suffix']  = options.cd_sim_suffix
            kwargs['exp_path']    = options.cd_exp_path
            kwargs['exp_prefix']  = options.cd_exp_prefix
            kwargs['exp_suffix']  = options.cd_exp_suffix
            kwargs['noise']       = options.cd_noise
            kwargs['in_pkl']      = options.cd_in_pkl
            kwargs['out_pkl']     = options.cd_out_pkl
            kwargs['data_weight'] = options.cd_data_weight
            kwargs['nmodels']     = options.nmodels
            kwargs['models_list'] = self.models_list
            self.observables['cd'] = cd_data.CD(kwargs)
        
        if 'deer' in self.experiments:
            kwargs = dict()
            kwargs['sim_path']    = options.deer_sim_path
            kwargs['sim_prefix']  = options.deer_sim_prefix
            kwargs['sim_suffix']  = options.deer_sim_suffix
            kwargs['exp_path']    = options.deer_exp_path
            kwargs['exp_prefix']  = options.deer_exp_prefix
            kwargs['exp_suffix']  = options.deer_exp_suffix
            kwargs['labels']      = options.deer_labels
            kwargs['moddepth']    = options.deer_moddepth
            kwargs['noise']       = options.deer_noise
            kwargs['data_weight'] = options.deer_data_weight
            kwargs['in_pkl']      = options.deer_in_pkl
            kwargs['in_hd5']      = options.deer_in_hd5
            kwargs['in_sim_pkl']  = options.deer_in_sim_pkl
            kwargs['in_sim_hd5']  = options.deer_in_sim_hd5
            kwargs['out_pkl']     = options.deer_out_pkl
            kwargs['nmodels']     = options.nmodels
            kwargs['models_list'] = self.models_list
            self.observables['deer']      = deer.Deer(kwargs)

        if 'scattering' in self.experiments:
            kwargs = dict()
            kwargs['sim_path']    = options.scattering_sim_path
            kwargs['sim_prefix']  = options.scattering_sim_prefix
            kwargs['sim_suffix']  = options.scattering_sim_suffix
            kwargs['exp_path']    = options.scattering_exp_path
            kwargs['exp_prefix']  = options.scattering_exp_prefix
            kwargs['exp_suffix']  = options.scattering_exp_suffix
            kwargs['noise']       = options.scattering_noise
            kwargs['scaling_factor']    = options.scattering_scaling_factor
            kwargs['additive_constant'] = options.scattering_additive_constant
            kwargs['solvent']     = options.scattering_solvent
            kwargs['data_weight'] = options.scattering_data_weight
            kwargs['in_pkl']      = options.scattering_in_pkl
            kwargs['in_hd5']      = options.scattering_in_hd5
            kwargs['in_sim_pkl']  = options.scattering_in_sim_pkl
            kwargs['in_sim_hd5']  = options.scattering_in_sim_hd5
            kwargs['out_pkl']     = options.scattering_out_pkl
            kwargs['nmodels']     = options.nmodels
            kwargs['models_list'] = self.models_list
            self.observables['scattering'] = scattering.Scattering(kwargs)

        self.nrestraints = get_nrestraints_all(self)
        self.exp = get_proc_exp(self)


    def get_proc_sim(self):
        """
        Converts simulated data to matrix by consideration of type of
        experimental data and noise of each data point.

        Parameters
        ----------
        self: object, contains all information about experimental data

        Returns
        -------
        sim, sim_init: matrices, simulated data points scaled by experimental noise
        """
        sim = np.zeros((self.nrestraints, len(self.models_list)))
        for j, nmodel in enumerate(self.models_list):
            sim_l = []
            for experiment in self.experiments:
                sim_tmp = self.observables[experiment].sim_tmp
                exp_err_tmp = self.observables[experiment].exp_err_tmp
                if experiment == 'deer':
                    for i, label in enumerate(self.observables['deer'].labels):
                        ln = "{}-{}".format(label[0], label[1])
                        moddepth = self.observables['deer'].moddepth[ln]
                        sim_l.extend((1 - moddepth + moddepth * sim_tmp[nmodel][ln]) /
                                      exp_err_tmp[ln])
                if experiment == 'scattering':
                    c = self.observables['scattering'].scaling_factor
                    sim_l.extend((c * sim_tmp[nmodel]) / exp_err_tmp)
                if experiment == 'generic':
                    sim_l.extend(sim_tmp[nmodel] / exp_err_tmp)
                if experiment == 'cd':
                    sim_l.extend(sim_tmp[nmodel] / exp_err_tmp)
            sim[:,j] = np.array(sim_l)
        return np.matrix(sim), np.matrix(sim.copy())


    def moddepth_fit(self, m, ln, wopt):
        """
        Fitting of the modulation depths for DEER data.

        Parameters
        ----------
        self: object, contains all information about experimental data
        m: float, moddepth of a single DEER trace
        ln: string, label name
        wopt: array-like, optimized/current weights

        Returns
        -------
        chi2: float, agreement of experimental and simulated data scaled by m
        """
        exp_label = np.asarray(self.observables['deer'].exp_tmp[ln][:,1])
        exp_err_label = self.observables['deer'].exp_err_tmp[ln]
        exp = ((np.array(exp_label, dtype=np.float64)) / exp_err_label)
        exp = np.matrix(np.array(exp).reshape(1, len(exp_err_label)))

        sim_label = np.zeros((len(exp_err_label), len(self.models_list)))
        for j, nmodel in enumerate(self.models_list):
            s = self.observables['deer'].sim_tmp[nmodel][ln]
            sim_label[:,j] = ((1 - m + m * s)/ exp_err_label)
        sim = np.matrix(sim_label)
        return optimize.common.chiSqrTerm(wopt, sim, exp)


    def coeff_fit(self, c, wopt):
        """
        Fit coefficient.
        """
        exp_err = self.observables['scattering'].exp_err_tmp
        exp = np.asarray(self.observables['scattering'].exp_tmp[:,1])
        exp = ((np.array(exp, dtype=np.float64)) / exp_err)
        exp = np.matrix(np.array(exp).reshape(1, len(exp_err)))

        sim_tmp = np.zeros((len(exp_err), len(self.models_list)))
        for j, nmodel in enumerate(self.models_list):
            s = self.observables['scattering'].sim_tmp[nmodel]
            sim_tmp[:,j] = ((c * s)/exp_err)
        sim = np.matrix(sim_tmp)
        return optimize.common.chiSqrTerm(wopt, sim, exp)


    def update_sim(self, wopt):
        """
        Updates sim and sim_init after optimization of nuisance parameters.

        Parameters
        ----------
        self: object, contains all information about experimental data
        wopt: array-like, optimized/current weights

        Returns
        -------
        sim, sim_init: matrices, simulated data points scaled by experimental noise
        """
        for experiment in self.experiments:
            if experiment == 'deer':
                for i, label in enumerate(self.observables['deer'].labels):
                    ln = "{}-{}".format(label[0], label[1])
                    m = self.observables['deer'].moddepth[ln]
                    m_opt, idx = leastsq(self.moddepth_fit, m, args=(ln, wopt))
                    self.observables['deer'].moddepth[ln] = m_opt

            if experiment == 'scattering':
                c = self.observables['scattering'].scaling_factor
                c_opt, idx = leastsq(self.coeff_fit, c, args=(wopt))
                self.observables['scattering'].scaling_factor = c_opt
        return self.get_proc_sim()


    def update_sim_init(self, wopt):
        if 'deer' in self.experiments:
            for label in self.observables['deer'].labels:
                ln = "{}-{}".format(label[0], label[1])
                if self.observables['deer'].moddepth[ln] == "initial-optimization":
                    self.observables['deer'].moddepth[ln] = 0.15
        if 'scattering' in self.experiments:
            if self.observables['scattering'].scaling_factor == "initial-optimization":
                self.observables['scattering'].scaling_factor = 0.0002
        return self.update_sim(wopt)



    def get_proc_sim_weights(self, wopt):
        """
        Provides weighted averages of the simulated data.

        Parameters
        ----------
        self: object, contains all information about experimental data
        wopt: array-like, optimized/current weights

        Returns
        -------
        sim_wopt: dictionary, contains all weighted simulated data
        """
        sim_wopt = dict()
        for experiment in self.experiments:
            if experiment == 'deer':
                sim_deer = dict()
                for i, label in enumerate(self.observables['deer'].labels):
                    ln = "{}-{}".format(label[0], label[1])
                    m = self.observables['deer'].moddepth[ln]
                    sim_label = np.zeros((len(self.observables['deer'].exp_err_tmp[ln]),
                                          len(self.models_list)))
                    for j, nmodel in enumerate(self.models_list):
                        s = self.observables['deer'].sim_tmp[nmodel][ln]
                        sim_label[:,j] = np.array(1 - m + m * s)
                    sim_deer[ln] = np.matrix(sim_label)*wopt
                sim_wopt['deer'] = sim_deer

            if experiment == 'scattering':
                c = self.observables['scattering'].scaling_factor
                sim_tmp = np.zeros((len(self.observables['scattering'].exp_err_tmp),
                                    len(self.models_list)))
                for j, nmodel in enumerate(self.models_list):
                    s = self.observables['scattering'].sim_tmp[nmodel]
                    sim_tmp[:,j] = np.array(c * s)
                sim_wopt['scattering'] = np.matrix(sim_tmp)*wopt

            if experiment == 'generic':
                sim_tmp_1 = dict()
                sim_tmp_dict = self.observables['generic'].sim_tmp_dict
                for i, flex_id in enumerate(self.observables['generic'].exp_tmp_dict.keys()):
                    sim_tmp_1[flex_id] = sim_tmp_dict[flex_id]*np.matrix(wopt)
                sim_wopt['generic'] = sim_tmp_1
            
            if experiment == 'cd':
                sim_tmp = np.zeros((len(self.observables['cd'].exp_err_tmp),
                                    len(self.models_list)))
                for j, nmodel in enumerate(self.models_list):
                    s = self.observables['cd'].sim_tmp[nmodel]
                    sim_tmp[:,j] = np.array(s)
                sim_wopt['cd'] = np.matrix(sim_tmp)*wopt
        return sim_wopt


def get_experiments(experiments):
    """
    Returns array of experimental techniques and
    checks if the provided kind of data is implemented
    in BioEn.
    """
    experiments_in_bioen = ['deer', 'scattering', 'generic', 'cd']
    if all(experiment.replace(" ", "") in experiments_in_bioen for experiment in experiments.split(',')):
        return experiments.replace(" ", "").split(',')
    else:
        for experiment in experiments.split(','):
            if experiment not in experiments_in_bioen:
                msg = 'ERROR: The experimental data type {} is not implemented in BioEn yet. ' + \
                      'Please try \"generic\" (useful for distances, NOEs, CS, J-couplings, ' + \
                      'PREs etc.)'.format(experiment)
                raise RuntimeError(msg)


def get_models_list(models_list_fn, nmodels):
    """
    Returns array of model ids.
    """
    return np.loadtxt(models_list_fn, unpack=True)[0:nmodels]


def get_nrestraints_all(self):
    """
    Returns total number of restraints (experimental data points).
    """
    nrestraints = 0
    for experiment in self.experiments:
        nrestraints += self.observables[experiment].nrestraints
    return nrestraints


def get_proc_exp(self):
    """
    Formats experimental data into a matrix.

    Parameters
    ----------
    self: object, contains all information about experimental data

    Returns
    -------
    exp: matrix, contains all experimental data in a matrix format
    """
    exp = []
    for experiment in self.experiments:
        exp_tmp = self.observables[experiment].exp_tmp
        exp_err_tmp = self.observables[experiment].exp_err_tmp
        if experiment == 'deer':
            for i, label in enumerate(self.observables['deer'].labels):
                ln = "{}-{}".format(label[0], label[1])
                exp.extend((np.array(exp_tmp[ln][:,2], dtype=np.float64)) / exp_err_tmp[ln])
        elif experiment == 'scattering':
            exp.extend((np.array(exp_tmp[:,1], dtype=np.float64)) / exp_err_tmp)
        elif experiment == 'generic':
            exp.extend((np.array(exp_tmp, dtype=np.float64)) / exp_err_tmp)
        elif experiment == 'cd':
            exp.extend((np.array(exp_tmp[:,2], dtype=np.float64)) / exp_err_tmp)
    return np.matrix(np.array(exp).reshape(1, self.nrestraints))
