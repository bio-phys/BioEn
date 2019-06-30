import sys
import numpy as np
from ... import utils
from .... import fileio


class CD:
    def __init__(self, kwargs):
        """
        Loads all information needed about data from circular dichroism (CD)
        experiments.

        Parameters
        ----------
        kwargs: provides options (settings) from user interface

        Returns
        -------
        cd: object, contains all information about jcouplings data
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

        if self.in_pkl is not None:
            [self.nrestraints, self.exp_tmp,
             self.exp_err_tmp, self.sim_tmp] = fileio.load(self.in_pkl)
        else:
            (self.nrestraints, self.exp_tmp, 
             self.exp_err_tmp) = get_exp_tmp(self)
            self.sim_tmp = get_sim_tmp(self)

            if self.out_pkl is not None:
                fileio.dump(self.out_pkl,
                            [self.nrestraints, self.exp_tmp,
                             self.exp_err_tmp, self.sim_tmp])


def get_exp_tmp(self):
    """
    Provides dictionary of experimental data.
    exp_tmp[:,0] --> wavelength
    exp_tmp[:,1] --> raw data
    exp_tmp[:,2] --> fit of raw data
    """
    exp_tmp = np.genfromtxt("{}/{}-{}-{}-{}.dat".format(self.exp_path, self.exp_prefix,
                                                    self.exp_suffix), comments="#")
    nrestraints += exp_tmp[:, 0].shape[0]

    if any(extension in self.noise[-4:] for extension in [".dat", ".txt"]):
        tmp = np.genfromtxt(self.nois), comments="#")
        wavelength_err_tmp = tmp[:,0]
        exp_err_tmp = tmp[:,1]
        # check if exp_tmp and exp_err_tmp have the same length of data
        if len(exp_tmp[:,0]) != len(exp_err_tmp):
            msg = "ERROR: Please check number of data points (number of lines) " +\
                  "in CD experimental data file and noise file (should be the same)". 
            raise RuntimeError(msg)
        # check if exp_tmp and exp_err_tmp have the same wavelength
        if np.sum(exp_tmp[:,0] - wavelength_err_tmp) != 0.0:
            msg = "ERROR: Please check wavelength content in CD experimental data " +\
                  "file and noise file (should be the same)". 
            raise RuntimeError(msg)
    elif self.noise == "exp_fit_dif":
        exp_err_tmp = np.array([np.abs(exp_tmp[:, 1] - exp_tmp[:, 2])])
    elif self.noise == "exp_fit_std":
        exp_err_tmp = np.array([np.std(exp_tmp[:, 1] - exp_tmp[:, 2])]*len(exp_tmp))
    elif utils.is_float(self.noise):
        exp_err_tmp = np.array([float(self.noise)]*len(exp_tmp))

    exp_err_tmp[exp_err_tmp == 0.0] = 0.01
    return (nrestraints, exp_tmp, exp_err_tmp)


def get_sim_tmp(self):
    """
    Provides dictionary of simulated data.
    """

    sim_tmp_1 = []
    sim_tmp_dict = dict()
    for flex_id in self.exp_tmp_dict.keys():
        fn = "{}/{}-{}-{}.dat".format(self.sim_path, self.sim_prefix, flex_id, self.sim_suffix)

        try:
            sim_generic = np.genfromtxt(fn, comments='#')
        except:
            print('ERROR: Cannot find simulated data file \'{}\''.format(fn))
            raise

        if len(sim_generic) != self.nmodels:
            msg = "ERROR: Number of data points in file \'{}\' ".format(fn) + \
                  "and number of models (--number_of_models {}) are not the same.".format(self.nmodels)
            raise RuntimeError(msg)

        sim_tmp_1.append(sim_generic)
        sim_tmp_dict[flex_id] = sim_generic

    sim_tmp_1 = np.array(sim_tmp_1)

    sim_tmp = dict()
    for nmodel in range(0, self.nmodels):
        sim_tmp[nmodel] = sim_tmp_1[:, nmodel]

    return sim_tmp, sim_tmp_dict
