#!/usr/bin/python
################################################
#
#   BioEn - bioen_scattering.py
#
################################################

import numpy as np
from scipy.interpolate import interp1d

def adapt_q(sim_single, exp):
    """ 
    Linear interpolation to generate new intensity array of simulated data 
    based on q-values determined by other (experimental) intensity 
    """
    f2 = interp1d(sim_single[:,0], sim_single[:,1], kind='cubic')
    xnew = exp[:,0]
    sim_single_new = np.zeros((len(exp[:,0]), 2))
    sim_single_new[:,0] = exp[:,0]
    sim_single_new[:,1] = f2(xnew)
    return sim_single_new
