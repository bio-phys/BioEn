#!/usr/bin/python
################################################
#
#   BioEn - bioen_rotamer.py
#
################################################

import numpy as np
import pickle
import pandas as pd

import MDAnalysis as mda
import MDAnalysis.analysis.align
import MDAnalysis.lib.NeighborSearch as KDNS
import MDAnalysis.analysis.distances

from scipy.special import fresnel

import matplotlib as m
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker


### preparation 

def fit_rotamers(rotamers, protein, resid, chainid, dcd_fn):
    """Produce a temporary trajectory of the rotamers.

    The backbone of the rotamers is fitted to the backbone of the
    spin labelled residues.
    """
    fittingSelection = (["name C", "name CA", "name N"],
                        ["protein and name C and resid {0} and segid {1}".format(resid, chainid),
                         "protein and name CA and resid {0} and segid {1}".format(resid, chainid),
                         "protein and name N and resid {0} and segid {1}".format(resid, chainid)])
    
    # fit the rotamer library onto the protein
    MDAnalysis.analysis.align.AlignTraj(rotamers, protein,
                                        select=fittingSelection, 
                                        weights="mass",
                                        filename=dcd_fn,
                                        verbose=True).run()
    return dcd_fn


def find_clashing_rotamers(clash_distance, fitted_rotamers, protein, resid, chainid):
    """Detect any rotamer that clashes with the protein."""
    
    # make a KD tree of the protein neighbouring atoms
    proteinNotSite = protein.select_atoms("protein and not name H* and not (resid " + str(resid) +
                                          " or (resid " + str(resid-1) + " and (name C or name O)) "
                                          "or (resid " + str(resid+1) + " and name N))")
    proteinNotSiteLookup = KDNS.AtomNeighborSearch(proteinNotSite)

    rotamerSel = fitted_rotamers.select_atoms("not name H*")

    rotamer_clash = []
    for rotamer in fitted_rotamers.trajectory:
        bumps = proteinNotSiteLookup.search(rotamerSel, clash_distance)
        rotamer_clash.append(bool(bumps))
    return rotamer_clash





def deer_ft(d, t, lambda_v=1.0, D_dip=2*np.pi*52.04*10**-3):
    """
    F_v/mu = C_F(x) cos(pi/6x^2) + S_F(x) sine(pi/6x^2)
    sin_f(x) = int_0^x sin(pi/2 t^2)
    cos_f(x) = int_o^x cos(pi/2 t^2)
    with x = [6 D_dip t / pi r^3]^0.5 and D_dip= 2pi 52.04 MHz nm^3
    
    Parameters
    -----------
    d: array
        Spin-spin distance [nm] for N structures
    t: array
        Time points [ns], M time points to be calculated
    D_dip: float
        2 pi 52.04 MHz nm^3 = 2 pi 52.04 nm^3/ns
    lambda_v: float
        Modulation depth
    
    Returns
    -------
    vt: array
        DEER/PELDOR time traces for each of the N structures for M time points (NxM)
    """
    x = (6 * D_dip *t/ (np.pi*d**3))**0.5
    
    # SciPy follows convention from Abramowitz and Stergun.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fresnel.html#s
    
    sin_f, cos_f = fresnel(x)
    F_vmu = (cos_f * np.cos(np.pi/6.*x**2) + sin_f * np.sin(np.pi/6.*x**2)) / x
    
    return (1. - lambda_v) + lambda_v* F_vmu


def get_ft(d, times):
    ft = []
    ft.append([0.0, 1.0])
    for t in times[1:]:
        ft.append([t/1000.0, deer_ft(d/10.0, t)])
    ft = np.array(ft)
    return ft


def get_deer_from_single_pair(slp, rotamer_labels, rotamer_library_weights, 
                              path_output_preparation):
    rl_weights = np.loadtxt(rotamer_library_weights)
       
    l0_resid = slp[0][0]
    l0_chainid = slp[0][1]
    l0_ln = '{}-{}'.format(l0_resid, l0_chainid)
    rotamer0_site = rotamer_labels[l0_ln]['site']
    rotamer0_clash = rotamer_labels[l0_ln]['clash']
    rotamer0_nitrogen = rotamer0_site.select_atoms("name N1")
    rotamer0_oxygen = rotamer0_site.select_atoms("name O1")
    
    l1_resid = slp[1][0]
    l1_chainid = slp[1][1]
    l1_ln = '{}-{}'.format(l1_resid, l1_chainid)
    rotamer1_site = rotamer_labels[l1_ln]['site']
    rotamer1_clash = rotamer_labels[l1_ln]['clash']
    rotamer1_nitrogen = rotamer1_site.select_atoms("name N1")
    rotamer1_oxygen = rotamer1_site.select_atoms("name O1")
    
    frames_dict = dict()
    sim_tmp = dict()
    win_dict = dict()
    count_all = 0
    for rotamer0 in rotamer0_site.trajectory:
        if not rotamer0_clash[rotamer0.frame]:
            for rotamer1 in rotamer1_site.trajectory:
                if not rotamer1_clash[rotamer1.frame]:
                    count_all += 1

                    (a, b, distance_nitrogen) = mda.analysis.distances.dist(rotamer0_nitrogen, 
                                                                            rotamer1_nitrogen)
                    (a, b, distance_oxygen) = mda.analysis.distances.dist(rotamer0_oxygen, 
                                                                          rotamer1_oxygen)
                    distance = np.mean([distance_nitrogen[0], distance_oxygen[0]])                    
                    
                    frames_dict[count_all] = {l0_resid: rotamer0.frame,
                                              l1_resid: rotamer1.frame,
                                              'dist': distance}
                    win_dict[count_all] = rl_weights[rotamer0.frame] * rl_weights[rotamer1.frame]

                    ft = get_ft(distance, slp[2])
                    sim_tmp[count_all] = {'{}-{}'.format(l0_resid, l1_resid): ft[:,1]}
    
    # data*.pkl contains all DEER traces according to a specific ensemble member
    data_pkl = "{}/data_{}_{}.pkl".format(path_output_preparation, l0_ln, l1_ln)                        
    pickle.dump([sim_tmp], open(data_pkl, "wb"))
    
    # models*.dat contains information for the nameing of the ensemble members
    frames = []
    win_tmp = []
    for frame in frames_dict.keys():
        frames.append(frame)
        win_tmp.append(win_dict[frame])
    models_dat = '{}/models_{}_{}.dat'.format(path_output_preparation, l0_ln, l1_ln)
    np.savetxt(models_dat, np.array(frames))
   
    # weights*.dat contains the prior weights (normalized) derived from the MC simulation 
    wsum = np.sum(win_tmp)
    win = np.array(win_tmp)/np.array(wsum)
    weights_dat = '{}/weights_{}_{}.dat'.format(path_output_preparation, l0_ln, l1_ln)
    np.savetxt(weights_dat, np.array(win))
   
    # frames*.dat contains information about each ensemble member being a combination 
    # of two rotamer states of the spin_labels and the distance between the spin-labels rotamers,
    # needed to backcalculate spin-label distribution of a labeled residue
    frames_pkl = "{}/frames_{}_{}.pkl".format(path_output_preparation, l0_ln, l1_ln)             
    pickle.dump([frames_dict], open(frames_pkl, "wb"))
      
    print 'Generated files are saved in {}:'.format(path_output_preparation)
    print '\t{} --> contains all calculated DEER traces based on spin-label distances'.format(data_pkl.split('/')[-1])
    print '\t{} --> list of the model ids'.format(models_dat.split('/')[-1])
    print '\t{} --> prior weights (normalized) derived from input weights of single rotamer states'.format(weights_dat.split('/')[-1])
    print '\t{} --> spin-label rotamers and spin-label distances for each ensemble member'.format(frames_pkl.split('/')[-1])
    print 'Number of rotamer combinations (number of ensemble members):', len(frames)
    
    return


def get_weighted_rotamer_states(frames_dict, bioen_data, label, theta=1000.0):

    nmodels = bioen_data[theta]["nmodels"]
    nmodels_weights = (np.vstack((bioen_data[theta]["nmodels_list"], 
                       np.array(bioen_data[theta]["wopt"]).reshape(1,-1)))).T
    
    rotamer_weights_1 = dict()
    for d in nmodels_weights:
        model = int(d[0])
        weight = d[1]
        r_new = frames_dict[model][label]
        
        if r_new in rotamer_weights_1:
            rotamer_weights_1[r_new].append(weight)
        elif r_new not in rotamer_weights_1:
            rotamer_weights_1[r_new] = [weight]
    
    rotamer_weights_2 = dict()
    for r_new, v in rotamer_weights_1.iteritems():
        rotamer_weights_2[r_new] = np.sum(v)
    
    return rotamer_weights_2


