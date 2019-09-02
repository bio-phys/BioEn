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
    exp_tmp = np.genfromtxt("{0}/{1}-{2}.dat".format(self.exp_path, self.exp_prefix,
                                                  self.exp_suffix), comments="#")
    nrestraints = exp_tmp[:, 0].shape[0]
    
    if any(extension in self.noise[-4:] for extension in [".dat", ".txt"]):
        tmp = np.genfromtxt(self.noise, comments="#")
        wavelength_err_tmp = tmp[:,0]
        exp_err_tmp = tmp[:,1]
        # check if exp_tmp and exp_err_tmp have the same length of data
        if len(exp_tmp[:,0]) != len(exp_err_tmp):
            msg = "ERROR: Please check number of data points (number of lines) " +\
                  "in CD experimental data file and noise file (should be the same)." 
            raise RuntimeError(msg)
        # check if exp_tmp and exp_err_tmp have the same wavelength
        if np.sum(exp_tmp[:,0] - wavelength_err_tmp) != 0.0:
            msg = "ERROR: Please check wavelength content in CD experimental data " +\
                  "file and noise file (should be the same)." 
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
    Provides dictionary of simulated CD data.
    """
    if len(self.models_list) != self.nmodels:
        msg = "ERROR: Number of models (--number_of_models {}) ".format(self.nmodels) +\
              "and number of IDs available in file '\{}\' ".format(len(self.models_list)) +\
              "are not the same."
        raise RuntimeError(msg)

    sim_tmp = dict()
    for model in self.models_list:
        sim_tmp_2 = np.genfromtxt("{0}/{1}{2}-{3}.dat".format(self.sim_path,
                                                                      self.sim_prefix, 
                                                                      int(model), 
                                                                      self.sim_suffix))
        if len(self.exp_tmp[:,0]) != len(sim_tmp_2[:,0]):
            msg = "ERROR: Please check number of data points (number of lines) " +\
                  "in CD experimental data file and simulated data file " +\
                  "(should be the same)." 
            raise RuntimeError(msg)
        # check if exp_tmp and exp_err_tmp have the same wavelength
        if np.sum(self.exp_tmp[:,0] - sim_tmp_2[:,0]) != 0.0:
            msg = "ERROR: Please check wavelength content in CD experimental data " +\
                  "file and CD simulated data file (should be the same)." 
            raise RuntimeError(msg)
        sim_tmp[model] = sim_tmp_2[:,1]

    return sim_tmp
