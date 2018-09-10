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
from matplotlib import gridspec 
import matplotlib.ticker as ticker


# preparation

def fit_rotamers(rotamers, protein, resid, chainid, dcd_fn):
    """
    Generates a temporary trajectory of the spin-label rotamers,
    which is attached to a spin-label position.
    
    Parameters
    ----------
    rotamers: trajectory of rotamer library
    protein: pdb
    resid: int
	spin-labeled position
    chainid: string
	chain of the spin-labeled position
    dcd_fn: string
	file name of temporary trajectory
    """
    fittingSelection = (["name C", "name CA", "name N"],
                        ["protein and name C and resid {0} and segid {1}".format(resid, chainid),
                         "protein and name CA and resid {0} and segid {1}".format(resid, chainid),
                         "protein and name N and resid {0} and segid {1}".format(resid, chainid)])
    
    # fit the rotamer library onto the protein according to the backbone
    # of the spin-labeled residue
    MDAnalysis.analysis.align.AlignTraj(rotamers, protein,
                                        select=fittingSelection, 
                                        weights="mass",
                                        filename=dcd_fn,
                                        verbose=True).run()
    return 


def find_clashing_rotamers(clash_distance, rotamer_site, protein, resid, chainid):
    """
    Detect any rotamer that clashes with the protein.
    
    Parameters
    ----------
    clash_distance: float
        distance between atoms of the spin-spin label and the protein
    rotamer_site: 
	trajectory of rotamer states attached to spin-label position
    protein: pdb
    resid: int
        spin-labeled position
    chainid: string
        chain of the spin-labeled position

    Returns
    -------
    rotamer_clash: array
        For each spin-label position, an array with boolean entries 
        of each rotameric state (True: clash, False: no clash)
    """
    
    # make a KD tree of the protein neighbouring atoms
    proteinNotSite = protein.select_atoms("protein and not name H* and not (resid " + str(resid) +
                                          " or (resid " + str(resid-1) + " and (name C or name O)) "
                                          "or (resid " + str(resid+1) + " and name N))")
    proteinNotSiteLookup = KDNS.AtomNeighborSearch(proteinNotSite)

    rotamerSel = rotamer_site.select_atoms("not name H*")

    rotamer_clash = []
    for rotamer in rotamer_site.trajectory:
        bumps = proteinNotSiteLookup.search(rotamerSel, clash_distance)
        rotamer_clash.append(bool(bumps))
    return rotamer_clash


def get_experimental_timesteps(path_experimental_data, label_pair):
    """
    Parameters
    ----------
        path_experimental_data
        label_pair: list
    
    Returns
    -------
    array: time_steps [ns] of a particular DEER measurement
    """
    fn = "{}/exp-{}-{}-signal-deer.dat".format(path_experimental_data, 
                                               label_pair[0], label_pair[1])
    return np.loadtxt(fn)[:,0]*1000


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
        2 pi 52.04 MHz nm^3 = 2 pi 52.04/1000 nm^3/ns
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
    """
    Calculates for each distances between two unpaired electron the 
    DEER trace
    """	
    ft = []
    ft.append([0.0, 1.0])
    for t in times[1:]: ft.append([t/1000.0, deer_ft(d/10.0, t)])
    ft = np.array(ft)
    return ft


def get_deer_from_single_pair(slp, rotamer_labels, rotamer_library_weights, 
                              path_output_preparation):
    """
    Samples over the tracetory of rotameric states, which are accepted due to
    steric clashes. Measures then the distances and calculates the DEER trace.
    Saves then the simulated information in the following files:
    data_*.pkl --> DEER traces based on spin-label distances
    models_*.dat --> list of the model ids
    frames_*.pkl --> spin-label rotamers and spin-label distances
    """	
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
    data_pkl = "{}/data_{}-{}.pkl".format(path_output_preparation, l0_resid, l1_resid)
    pickle.dump([sim_tmp], open(data_pkl, "wb"))
    
    # models*.dat contains information for the nameing of the ensemble members
    frames = []
    win_tmp = []
    for frame in frames_dict.keys():
        frames.append(frame)
        win_tmp.append(win_dict[frame])
    models_dat = '{}/models_{}-{}.dat'.format(path_output_preparation, l0_resid, l1_resid)
    np.savetxt(models_dat, np.array(frames))
   
    # weights*.dat contains the prior weights (normalized) derived from the MC simulation 
    wsum = np.sum(win_tmp)
    win = np.array(win_tmp)/np.array(wsum)
    weights_dat = '{}/weights_{}-{}.dat'.format(path_output_preparation, l0_resid, l1_resid)
    np.savetxt(weights_dat, np.array(win))
   
    # frames*.dat contains information about each ensemble member being a combination 
    # of two rotamer states of the spin_labels and the distance between the spin-labels rotamers,
    # needed to backcalculate spin-label distribution of a labeled residue
    frames_pkl = "{}/frames_{}-{}.pkl".format(path_output_preparation, l0_resid, l1_resid)             
    pickle.dump([frames_dict], open(frames_pkl, "wb"))
      
    print('Generated files are saved in {}:'.format(path_output_preparation))
    print('\t{} --> DEER traces based on spin-label distances'.format(data_pkl.split('/')[-1]))
    print('\t{} --> list of the model ids'.format(models_dat.split('/')[-1]))
    print('\t{} --> spin-label rotamers and spin-label distances'.format(frames_pkl.split('/')[-1]))
    return


# analysis

def get_weighted_rotamer_states(frames_data_lp, 
				bioen_data_lp, 
				label, theta=1000.0):
    """
    Margenalize rotamer distribution for each spin-label derived from BioEn 
    reweighting and possible rotameric states saved in frames*.pkl.

    Parameters
    -----------
    frames_data_lp: dict
        Rotameric states of each spin-label position for a certain spin-label
        pair
    bioen_data_lpt: dict
        BioEn result for a spin-label pair
    label: string
        Spin-label 
    theta: float
        Good confidence value
    
    Returns
    -------
    rotamer_weights_2: dict
        For each spin-label rotamer state the margenalized weights are given
    """
    nmodels = bioen_data_lp[theta]["nmodels"]
    nmodels_weights = (np.vstack((bioen_data_lp[theta]["nmodels_list"], 
                       np.array(bioen_data_lp[theta]["wopt"]).reshape(1,-1)))).T
    
    rotamer_weights_1 = dict()
    for d in nmodels_weights:
        model = int(d[0])
        weight = d[1]
        r_new = frames_data_lp[model][label]
        
        if r_new in rotamer_weights_1:
            rotamer_weights_1[r_new].append(weight)
        elif r_new not in rotamer_weights_1:
            rotamer_weights_1[r_new] = [weight]
    
    rotamer_weights_2 = dict()
    for r_new, v in rotamer_weights_1.iteritems():
        rotamer_weights_2[r_new] = np.sum(v)
    
    return rotamer_weights_2


def plot_deer_traces(bioen_data):
    """
    Visalize all experimental and ensemble averaged DEER traces 
    (before reweighting = X-ray; after reweighting = BioEn).
    """
    fs = 16
    fig = plt.figure(figsize=[16,6.0])
    plt.subplots_adjust(hspace = .001)
    gs = gridspec.GridSpec(3, 7)
    gs.update(wspace=0.0, hspace=0.0)
    
    axs = []
    for i in range(21):
        axs.append(plt.subplot(gs[i]))

    def plot_traces(ident_all, idx):
        label_id = ident_all[0]
        label_name = ident_all[1]
        theta = 1000.0
        theta_max = np.max(bioen_data[label_id].keys())
        label = bioen_data[label_id][theta]['exp']['deer'].keys()[0]
        exp = bioen_data[label_id][theta]['exp']['deer'][label]
        sim_init = bioen_data[label_id][theta_max]['sim_init']['deer'][label]
        sim = bioen_data[label_id][theta]['sim_wopt']['deer'][label]
        axs[idx].plot(exp[:,0], exp[:,1], color='black', linewidth=2)
        axs[idx].plot(exp[:,0], sim_init, color='green', linewidth=2)
        axs[idx].plot(exp[:,0], sim, color='red', linewidth=2)
        
        m.rcParams['axes.unicode_minus'] = False
        
        npoints = len(bioen_data[label_id][theta]['exp']['deer'][label][:,0])
        chi2_xray = bioen_data[label_id][theta_max]['chi2_init'] / float(npoints)
        chi2_bioen = bioen_data[label_id][theta]['chi2'] / float(npoints)

        if idx == 9:
            x_xray = 1.2
            y_xray = 0.9
            x_bioen = 1.2
            y_bioen = 0.75  
        elif idx in [10,13,17]:
            x_xray = 1.5
            y_xray = 0.6
            x_bioen = 1.5
            y_bioen = 0.47
        elif idx == 20:
            x_xray = 1.5
            y_xray = 0.85
            x_bioen = 1.5
            y_bioen = 0.74                
        else:
            x_xray = 1.0
            y_xray = 0.85
            x_bioen = 1.0
            y_bioen = 0.7
        
        if idx == 0:
            x_lp = 0.14
            y_lp = 0.34
        elif idx in [6,13,20]:
            x_lp = 3.3
            y_lp = 0.34            
        else:
            x_lp = 0.2
            y_lp = 0.34
        axs[idx].text(x_xray, y_xray, 
		      r"$\chi^2_{\mathrm{X-ray}}={%.2f}$".replace("-", u"\u2212") % chi2_xray, 
		      color='green', fontsize=14)
        axs[idx].text(x_bioen, y_bioen, r"$\chi^2_{\mathrm{BioEn}}={%.2f}$" % chi2_bioen, 
		      color='red', fontsize=14)
        axs[idx].text(x_lp, y_lp, label_id, fontsize=14)
        return chi2_xray, chi2_bioen
        
    idents = [["319-265","N265-A319"],["344-265","N265-E344"],["370-265","N265-V370"],
              ["", ""],               ["319-429","A319-Q429"],["319-460","A319-V460"],
	      ["259-457","Q259-V457"],["319-259","Q259-A319"],["344-259","Q259-E344"],
	      ["370-259","Q259-V370"],["374-259","Q259-Q374"],["344-429","E344-Q429"],
	      ["344-460","E344-V460"],["259-448","Q259-L448"],["319-292","I292-A319"],
	      ["344-292","I292-E344"],["370-292","I292-V370"],["374-292","I292-Q374"],
              ["370-429","V370-Q429"],["370-460","V370-V460"],["292-460","I292-V460"]]
    
    chi2_xray_all = []
    chi2_bioen_all = []
    for idx, ident_all in enumerate(idents):
        if ident_all[0] != "": 
            chi2_xray, chi2_bioen = plot_traces(ident_all, idx)
            chi2_xray_all.append(chi2_xray)
            chi2_bioen_all.append(chi2_bioen)
                
    print("chi2_xray_average", np.mean(chi2_xray_all))
    print("chi2_bioen_average", np.mean(chi2_bioen_all))
    
    # x-axis
    for i in [0,1,2,4,5,7,8,9,10,11,12,14,15,16,17]:
        axs[i].set_xticks(range(0,8,1))
        axs[i].set_xticklabels([])
        axs[i].set_xlim(0,3.9)

    for i in range(14,20,1):
        axs[i].set_xticks(range(0,8,1))
        axs[i].set_xticklabels(range(0,8,1), fontsize=fs)
        axs[i].set_xlabel(r't [$\mu$s]', fontsize=fs)
        axs[i].set_xlim(0,3.9)
    
    for i in [6,13,20]:
        axs[i].set_xticks(range(0,8,2))
        axs[i].set_xticklabels([])
        axs[i].set_xlim(0,6.9)
    axs[20].set_xticklabels(range(0,8,2), fontsize=fs)
    axs[20].set_xlabel(r't [$\mu$s]', fontsize=fs)
    axs[i].set_xlim(0,6.9)
    
    # y-axis
    for i in range(0,21):
        axs[i].set_yticks(np.arange(0.2,1.2,0.2))
        axs[i].set_yticklabels([], fontsize=fs)
        axs[i].set_ylim(0.3, 1.0)
        
    for i in [0,7,14]:
        axs[i].set_yticklabels(np.arange(0.2,1.2,0.2), fontsize=fs)
        axs[i].set_ylabel(r'F(t)', fontsize=fs)

    axs[3].spines['top'].set_visible(False)        
    axs[3].set_yticks([])        
    
    for i in range(0,21,1):
        axs[i].tick_params(direction='in')
    
    plt.tight_layout()
    plt.savefig("deer_all.png".format(idx), dpi=400)
    plt.show()
    return


def plot_reweighted_elbow(bioen_data):
    fs = 16
    fig = plt.figure(figsize=[16,6.0])
    plt.subplots_adjust(hspace = .001)
    gs = gridspec.GridSpec(3, 7)
    gs.update(wspace=0.0, hspace=0.0)
    
    axs = []
    for i in range(21):
        axs.append(plt.subplot(gs[i]))

    def plot_traces(ident_all, idx):
        label_id = ident_all[0]
        label_name = ident_all[1]
        theta = 1000.0
        theta_max = np.max(bioen_data[label_id].keys())
        label = bioen_data[label_id][thetas[0]]['exp']['deer'].keys()[0]
        npoints = len(bioen_data[label_id][theta]['exp']['deer'][label][:,0])
        
        g_all = []
        for i, theta in enumerate(thetas[::-1]):
            s = d[theta]['S']
            chi2 = bioen_data[label_id]['chi2'] / float(npoints)
            g = chi2 - theta*s
            if theta == 1000: 
                c=red1
                zo=2
            else: 
                c='black'
                zo=1
            axs[idx].scatter(-s, chi2, marker='.', s=150, color=c, edgecolor=c, zorder=zo)
            g_all.append([chi2, theta, s])
        g_all = np.array(g_all)
        axs[idx].plot(-g_all[:,2], g_all[:,0], color='black', zorder=0)
        
        x_lp = 2.6
        y_lp = 11.4
        axs[idx].text(x_lp, y_lp, label_id, fontsize=14)
        return
 
    idents = [["319-265","N265-A319"],["344-265","N265-E344"],["370-265","N265-V370"],
              ["", ""],               ["319-429","A319-Q429"],["319-460","A319-V460"],
	      ["259-457","Q259-V457"],["319-259","Q259-A319"],["344-259","Q259-E344"],
	      ["370-259","Q259-V370"],["374-259","Q259-Q374"],["344-429","E344-Q429"],
	      ["344-460","E344-V460"],["259-448","Q259-L448"],["319-292","I292-A319"],
	      ["344-292","I292-E344"],["370-292","I292-V370"],["374-292","I292-Q374"],
              ["370-429","V370-Q429"],["370-460","V370-V460"],["292-460","I292-V460"]]
    
    for idx, ident_all in enumerate(idents):
        if ident_all[0] != "": plot_traces(ident_all, idx)
    
    # x-axis
    for i in range(0,21):
        axs[i].set_xticks(range(-2,8,2))   
        axs[i].set_xticklabels([])
        axs[i].set_xlim(-0.4,6.5,)

    for i in range(14,21):
        axs[i].set_xticklabels(range(-2,8,2), fontsize=fs)
        axs[i].set_xlabel(r'$S_\mathrm{KL}$', fontsize=fs)
        axs[i].set_xlim(-0.4,6.5)
    
    # y-axis
    for i in range(0,21):
        axs[i].set_yticks(np.arange(0,100,4))   
        axs[i].set_yticklabels([])
        axs[i].set_ylim(-1,14)
        
    for i in [0,7,14]:
        axs[i].set_yticklabels(np.arange(0,100,4), fontsize=fs)
        axs[i].set_ylabel(r'$\chi^2$', fontsize=fs)
        axs[i].set_ylim(-1, 14)
      
    axs[3].spines['top'].set_visible(False)        
    axs[3].set_yticks([])        
    
    for i in range(0,21,1):
        axs[i].tick_params(direction='in')
    
    plt.tight_layout()
    plt.savefig("deer_all_singles_equal_elbow.png".format(idx), dpi=400)
    plt.show()
    return








