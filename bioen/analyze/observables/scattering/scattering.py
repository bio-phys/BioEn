from __future__ import print_function
import sys
import os
import numpy as np
from ... import utils
from .... import fileio


class Scattering:
    def __init__(self, kwargs):
        """
        Loads all information needed about Scattering data.

        Parameters
        ----------
        kwargs: provides options (settings) from user interface

        Returns
        -------
        scattering: object, contains all information about DEER data
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # TODO: unify input file names in_pkl and in_hd5

        # all input data in pkl format
        if self.in_pkl is not None:
            [self.coeff, self.nrestraints, self.exp_tmp,
             self.exp_err_tmp, self.sim_tmp] = fileio.load(self.in_pkl)
        # all input data in hd5 format
        elif self.in_hd5 is not None:
            # with open(self.in_hd5, 'r') as fp:
            #     [self.nrestraints, self.exp_tmp,
            #      self.exp_err_tmp, self.sim_tmp] = h5py.File(fp, 'r')
            [self.coeff, self.nrestraints, self.exp_tmp,
             self.exp_err_tmp, self.sim_tmp] = fileio.load(self.in_hd5,
                hdf5_keys=["coeff", "nrestraints", "exp_tmp", "exp_err_tmp", "sim_tmp"])
        # input data provided in different files
        else:
            self.coeff = get_coeff(self.coeff)
            self.nrestraints, self.exp_tmp, self.exp_err_tmp = get_exp_tmp(self)
            # simulated data provided in pkl format
            if self.in_sim_pkl is not None:
                [self.sim_tmp] = fileio.load(self.in_sim_pkl)
            # simulated data provided in hd5 data
            elif self.in_sim_hd5 is not None:
                [self.sim_tmp] = fileio.load(self.in_sim_hd5, hdf5_keys=["sim_tmp"])
            else:
                self.sim_tmp = get_sim_tmp(self)

        # TODO: unify output file names in_pkl and in_hd5

        # save data as output pkl and use it for the next run
        if self.out_pkl is not None:
            fileio.dump(self.out_pkl,
                        [self.nrestraints, self.exp_tmp,
                         self.exp_err_tmp, self.sim_tmp])
        # elif self.out_hd5 is not None:
        #     fileio.dump(self.out_hd5,
        #                 {"nrestraints": self.nrestraints,
        #                  "exp_tmp": self.exp_tmp,
        #                  "exp_err_tmp": self.exp_err_tmp,
        #                  "sim_tmp": self.sim_tmp})


def get_coeff(coeff):
    """
    Parameters
    ----------
    coeff: string,
        value for nuisance parameter or \"initial-optimization\"

    Returns
    -------
    coeff_new: float,
        value for nusicane parameter from input of after initial optimization
    """

    if coeff == "initial-optimization":
        coeff_new = "initial-optimization"
    else:
        try:
            coeff_new = float(self.coeff)
        except ValueError:
            print("Please provide information on the value of the nuisance parameter: "
                  "either single value (float) or let BioEn perform an initial "
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

    try:
        exp_1 = np.genfromtxt(fn, comments='#')
    except:
        print('ERROR: Cannot open file with simulated scattering data \'{}\''.format(fn))
        raise

    exp_tmp = exp_1[:, 0:2]
    exp_err_tmp = exp_1[:, 2]

    nrestraints = len(exp_tmp)
    return nrestraints, exp_tmp, exp_err_tmp


def get_sim_tmp(self):
    """
    Load simulated data for each model.
    """
    sim_tmp = dict()

    for nmodel in range(0, self.nmodels):
        fn = "{}/{}{}-{}.dat".format(self.sim_path, self.sim_prefix, nmodel, self.sim_suffix)

        try:
            sim_tmp[nmodel] = np.genfromtxt(fn, comments='#')[:, 1]
        except:
            print('ERROR: Cannot open simulated data scattering file \'{}\''.format(fn))
            raise

    return sim_tmp
