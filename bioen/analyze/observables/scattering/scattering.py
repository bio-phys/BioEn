from __future__ import print_function
import sys
if sys.version_info >= (3,):
    import pickle
else:
    import cPickle as pickle
import os
import numpy as np
# disable FutureWarning, intended to warn H5PY developers, but may confuse our users
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
from ... import utils


class Scattering:
    def __init__(self, kwargs):
        """
        Loads all information needed about Scattering data.

        Parameters
        -----------
        kwargs: provides options (settings) from user interface

        Returns
        --------
        scattering: object, contains all information about DEER data
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # all input data in pkl format
        if self.in_pkl is not None:
            with open(self.in_pkl, 'r') as file:
                [self.coeff, self.nrestraints, self.exp_tmp,
                 self.exp_err_tmp, self.sim_tmp] = pickle.load(file)
        # all input data in hd5 format
        elif self.in_hd5 is not None:
            with open(self.in_hd5, 'r') as file:
                [self.nrestraints, self.exp_tmp,
                 self.exp_err_tmp, self.sim_tmp] = h5py.File(file, 'r')
        # input data provided in different files
        else:
            self.coeff = get_coeff(self.coeff)
            self.nrestraints, self.exp_tmp, self.exp_err_tmp = get_exp_tmp(self)
            # simulated data provided in pkl format
            if self.in_sim_pkl is not None:
                with open(self.in_sim_pkl, 'r') as file:
                    [self.sim_tmp] = pickle.load(file)
            # simulated data provided in hd5 data
            elif self.in_sim_hd5 is not None:
                with open(self.in_sim_hd5, 'r') as file:
                    [self.sim_tmp] = h5py.File(file, 'r')
            else:
                self.sim_tmp = get_sim_tmp(self)

        # save data as output pkl and use it for the next run
        if self.out_pkl is not None:
            pickle.dump([self.nrestraints, self.exp_tmp,
                         self.exp_err_tmp, self.sim_tmp], open(self.out_pkl, 'wb'))


def get_coeff(coeff):
    """
    Parameters
    -----------
    coeff: string,
        value for nuisance parameter or \"initial-optimization\"

    Returns
    --------
    coeff_new: float,
        value for nusicane parameter from input of after initial optimization
    """

    if coeff == "initial-optimization":
        coeff_new = "initial-optimization"
    else:
        try:
            coeff_new = float(self.coeff)
        except ValueError:
            print("Please provide information on the value of the nuisance parameter: " \
                  "either single value (float) or let BioEn perform an initial " \
                  "optimization of the nuiscance parameter.")
    return coeff_new


def get_exp_tmp(self):
    """
    Provides dictionary of experimental data.
    exp_tmp[:,0] --> q
    exp_tmp[:,1] --> I(q)
    exp_tmp[:,2] --> noise
    """

    fn = "{}/{}-{}.dat".format(self.exp_path, self.exp_prefix, self.exp_suffix)
    exp_1 = np.genfromtxt(fn, comments='#')

    exp_tmp = exp_1[:, 0:2]
    exp_err_tmp = exp_1[:, 2]

    nrestraints = len(exp_tmp)
    return nrestraints, exp_tmp, exp_err_tmp


def get_sim_tmp(self):
    """
    xxxx
    """
    sim_tmp = dict()
    for nmodel in range(0, self.nmodels):
        fn = "{}/{}{}-{}.dat".format(self.sim_path, self.sim_prefix, nmodel, self.sim_suffix)
        sim_tmp[nmodel] = np.genfromtxt(fn, comments='#')[:, 1]
    return sim_tmp
