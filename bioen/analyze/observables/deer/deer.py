from __future__ import print_function
import sys
import numpy as np
from ... import utils
from .... import fileio


class Deer:
    def __init__(self, kwargs):
        """
        Loads all information needed about DEER data.

        Parameters
        ----------
        kwargs: provides options (settings) from user interface

        Returns
        -------
        deer: object, contains all information about DEER data
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

        if self.in_pkl is not None:
            [self.labels, self.moddepth, self.nrestraints, self.exp_tmp,
                self.exp_err_tmp, self.sim_tmp] = fileio.load(self.in_pkl)
        else:
            self.labels = get_labels(self.labels)
            self.moddepth = get_moddepth(self.moddepth, self.labels)
            self.nrestraints, self.exp_tmp, self.exp_err_tmp = get_exp_tmp(self)
            if self.in_sim_pkl is not None:
                [self.sim_tmp] = fileio.load(self.in_sim_pkl)
            else:
                self.sim_tmp = get_sim_tmp(self)

        # save data as output pkl and use it for the next run
        if self.out_pkl is not None:
            fileio.dump(self.out_pkl,
                        [self.labels, self.moddepth, self.nrestraints,
                         self.exp_tmp, self.exp_err_tmp, self.sim_tmp])


def get_labels(labels):
    """
    Provides array of label pairs.
    """
    labels_list = []
    for label in labels.split(','):
        labels_list.append([int(label.split('-')[0]), int(label.split('-')[1])])
    return labels_list


def get_moddepth(moddepth, labels):
    """
    Converts provided theta information to array.

    Parameters
    ----------
    moddepth: float, list of floats or file
    labels: array, label pairs

    Returns
    -------
    moddepth_new: dict, contains for each label pair a modulation depth
    """
    moddepth_new = dict()
    if utils.is_float(moddepth):
        for label in labels:
            ln = "{}-{}".format(label[0], label[1])
            moddepth_new[ln] = float(moddepth)
    elif '.dat' in moddepth:
        lines = utils.load_lines(moddepth)
        for le in lines:
            moddepth_new[le[0]] = float(le[1])
        for label in labels:
            ln = "{}-{}".format(label[0], label[1])
            if ln not in moddepth_new.keys():
                raise ValueError('Please provide modulation depth of label \'{}\' '.format(ln) +
                                 'in file \'{}\'.'.format(moddepth))
    elif 'initial-optimization' == moddepth:
        for label in labels:
            ln = "{}-{}".format(label[0], label[1])
            moddepth_new[ln] = 'initial-optimization'
    else:
        raise ValueError('Please provide theta value(s) is appropriate format.')

    return moddepth_new


def get_exp_tmp(self):
    """
    Provides dictionary of experimental data.
    exp_tmp[:,0] --> time points of the measurement
    exp_tmp[:,1] --> background corrected data
    exp_tmp[:,2] --> fit of background corrected data
    """
    exp_tmp = dict()
    exp_err_tmp = dict()
    nrestraints = 0
    for i, label in enumerate(self.labels):
        ln = "{}-{}".format(label[0], label[1])
        tmp = np.genfromtxt("{}/{}-{}-{}-{}.dat".format(self.exp_path, self.exp_prefix,
                                                        label[0], label[1],
                                                        self.exp_suffix), comments="#")
        exp_tmp[ln] = tmp
        nrestraints += tmp[:, 0].shape[0]

        if any(extension in self.noise[-4:] for extension in [".dat", ".txt"]):
            for line in utils.load_lines(self.noise):
                if line[0] == ln: tmp_2 = np.array([float(line[1])]*len(tmp))
            try:
               tmp_2
            except:
               msg = "ERROR: Missing noise value of spin-label pair \'{}\' ".format(ln) +\
                     "in file \'{}\'.".format(self.noise)
               raise RuntimeError(msg)

        elif self.noise == "exp_fit_dif":
            tmp_2 = np.array([np.abs(tmp[:, 1] - tmp[:, 2])])[0]
        elif self.noise == "exp_fit_std":
            tmp_2 = np.array([np.std(tmp[:, 1] - tmp[:, 2])]*len(tmp))
        elif utils.is_float(self.noise):
            tmp_2 = np.array([float(self.noise)]*len(tmp))

        try:
            tmp_2
        except:
            msg = "ERROR: Please provide the correct format for DEER noise. " +\
                  "Current format: \'{}\'".format(self.noise)
            raise RuntimeError(msg)

        tmp_2[tmp_2 == 0.0] = 0.01
        exp_err_tmp[ln] = tmp_2
    return nrestraints, exp_tmp, exp_err_tmp


def get_sim_tmp(self):
    """
    Provides dictionary of simulated data (simulated DEER trace).
    """
    if len(self.models_list) != self.nmodels:
        print("ERROR: Number of models (--number_of_models {}) ".format(self.nmodels) +
              "and number of IDs available in file '\{}\' ".format(len(self.models_list)) +
              "are not the same.")
    sim_tmp = dict()
    for model in self.models_list:
        sim_tmp_2 = dict()
        for i, label in enumerate(self.labels):
            ln = "{}-{}".format(label[0], label[1])
            sim_tmp_2[ln] = np.genfromtxt("{0}/{1}{2}-{3}-{4}-{5}.dat".format(self.sim_path,
                                                                              self.sim_prefix, 
                                                                              int(model), 
                                                                              label[0], 
                                                                              label[1], 
                                                                              self.sim_suffix))[:, 1]
        sim_tmp[model] = sim_tmp_2
    return sim_tmp
