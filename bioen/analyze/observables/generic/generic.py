import sys
import numpy as np
from ... import utils
from .... import fileio


class Generic:
    def __init__(self, kwargs):
        """
        Loads all information needed about data from different sources
        (e.g. NOEs, distances).

        Parameters
        ----------
        kwargs: provides options (settings) from user interface

        Returns
        -------
        generic: object, contains all information about jcouplings data
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

        if self.in_pkl is not None:
            [self.data_ids, self.nrestraints, self.exp_tmp,
             self.exp_tmp_dict, self.exp_err_tmp, self.exp_err_tmp_dict,
             self.sim_tmp] = fileio.load(self.in_pkl)
        else:
            self.data_ids = get_data_ids(self.data_ids)
            (self.nrestraints, self.exp_tmp, self.exp_tmp_dict,
                self.exp_err_tmp, self.exp_err_tmp_dict) = get_exp_tmp(self)
            self.sim_tmp, self.sim_tmp_dict = get_sim_tmp(self)

            if self.out_pkl is not None:
                fileio.dump(self.out_pkl,
                            [self.data_ids, self.nrestraints, self.exp_tmp,
                                self.exp_tmp_dict, self.exp_err_tmp, self.exp_err_tmp_dict,
                                self.sim_tmp])


def get_data_ids(data_ids):
    """
    Provides array of ids for data from different measurements.
    """
    return data_ids.split(',')


def get_exp_tmp(self):
    """
    Provides dictionary of experimental data.
    """

    fn = "{}/{}-{}.dat".format(self.exp_path, self.exp_prefix, self.exp_suffix)
    try:
        lines = utils.load_lines(fn)
    except IOError:
        print('ERROR: Cannot find experimental data file \'{}\''.format(fn))

    exp_tmp = []
    exp_tmp_dict = dict()
    exp_err_tmp = []
    exp_err_tmp_dict = dict()
    for le in lines:
        flex_id_tmp = le[0]
        if (len(self.data_ids) == 1 and self.data_ids[0] == 'all') or \
                (flex_id_tmp in self.data_ids):
            exp_tmp.append(float(le[1]))
            exp_tmp_dict[flex_id_tmp] = float(le[1])
            exp_err_tmp.append(float(le[2]))
            exp_err_tmp_dict[flex_id_tmp] = float(le[2])
        else:
            print("WARNING: Experimental data ID \'{}\' is not used in ".format(flex_id_tmp) +
                  "reweighting, since it is undefined in the submission script. " +
                  "If you are ok with this, you can ignore this warning.")
    nrestraints = len(exp_tmp_dict.keys())
    return (nrestraints, exp_tmp, exp_tmp_dict, exp_err_tmp, exp_err_tmp_dict)


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
