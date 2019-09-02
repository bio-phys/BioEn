import sys
import numpy as np
import pandas as pd

import matplotlib as m
# m.use('qt5agg')
m.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Show_plot:
    def __init__(self, bioen_pkl, simple_plot_output):
        """
        Visualizes experimental data with ensemble averaged data.
        """
        self.bioen_pkl = bioen_pkl
        self.simple_plot_output = simple_plot_output
        self.bioen_data = load_bioen_data(self.bioen_pkl)
        start_plotting(self)


def load_bioen_data(bioen_pkl):
    """
    Loads bioen output.

    Parameters
    ----------
    bioen_pkl: string,
        file name of bioen run

    Returns
    -------
    bioen_data: pkl
        pkl file contains all information about bioen run
    """
    df = pd.read_pickle(bioen_pkl)
    bioen_data = df.to_dict()

    return bioen_data


def start_plotting(self):
    """
    Based on experimental data defined in bioen_data, decide on
    the type of the plot.

    Parameters
    ----------
    self: object, contains bioen_data

    """
    pp = PdfPages(self.simple_plot_output)
    experiments = self.bioen_data[np.max(list(self.bioen_data.keys()))]['exp'].keys()
    theta_series = np.sort(list(self.bioen_data.keys()))[::-1]
    figs_all = []

    if 'deer' in experiments:
        figs_all.append(plot_deer(self, theta_series))
    if 'scattering' in experiments:
        figs_all.append(plot_scattering(self, theta_series))
    if 'generic' in experiments:
        figs_all.append(plot_generic(self, theta_series))
    if 'cd' in experiments:
        figs_all.append(plot_cd(self, theta_series))
    figs_all = [fig for sublist in figs_all for fig in sublist]
    for fig in figs_all:
        pp.savefig(fig)
    pp.close()
    return


def plot_deer(self, theta_series):
    theta_max = np.max(theta_series)
    figs = []
    for label in self.bioen_data[theta_max]['exp']['deer'].keys():
        figs.append(visualize_deer_traces_all_thetas(self.bioen_data, label,
                                                     theta_series))
    return figs


def visualize_deer_traces_all_thetas(bioen_data, label, theta_series):
    fig = plt.figure(figsize=[9, 6])
    ax = fig.add_subplot(111)

    exp = bioen_data[np.max(theta_series)]['exp']['deer'][label]
    ax.plot(exp[:, 0], exp[:, 1], color='black', linewidth=2.5, label='Exp.', zorder=2)

    a = np.linspace(0.1, 0.7, num=len(theta_series))
    for i, theta in enumerate(theta_series):
        sim = bioen_data[theta]['sim_wopt']['deer'][label]
        ax.plot(exp[:, 0], sim, color='red', alpha=a[i], linewidth=3.0,
                label=r"BioEn ($\theta={}$)".format(theta), zorder=3)

    ax.set_xlabel(r't [$\mu$s]')
    ax.set_ylabel(r'F(t)')
    ax.legend(loc=1, ncol=2)
    plt.grid()
    plt.tight_layout()
    return fig


def plot_scattering(self, theta_series):
    fig = plt.figure(figsize=[9, 6])
    ax = fig.add_subplot(111)

    exp = self.bioen_data[np.max(theta_series)]['exp']['scattering']
    ax.plot(exp[:, 0], exp[:, 1], color='black', linewidth=2.5, label='Exp.', zorder=2)

    a = np.linspace(0.1, 0.7, num=len(theta_series))
    for i, theta in enumerate(theta_series):
        sim = self.bioen_data[theta]['sim_wopt']['scattering']
        ax.plot(exp[:, 0], sim, color='red', alpha=a[i], linewidth=3.0,
                label=r"BioEn ($\theta={}$)".format(theta), zorder=3)

    ax.set_yscale('log')
    ax.set_xlabel(r'q [${\AA^{-1}}$]')
    ax.set_ylabel(r'I(q)')
    ax.legend(loc=1, ncol=2)
    plt.grid()
    plt.tight_layout()
    return [fig]


def plot_generic(self, theta_series):
    theta_max = np.max(theta_series)
    figs = []
    exp_keys = []
    for exp_key in self.bioen_data[theta_max]['exp']['generic'].keys():
        exp_keys.append(exp_key)
        if len(exp_keys) == 10:
            figs.append(visualize_generic_data(self.bioen_data,
                                               exp_keys, theta_series))
            exp_keys = []
    if len(exp_keys) < 10:
        figs.append(visualize_generic_data(self.bioen_data,
                                           exp_keys, theta_series))
    return figs


def visualize_generic_data(bioen_data, exp_keys, theta_series):
    fig = plt.figure(figsize=[12, 5])
    ax = fig.add_subplot(111)

    exp_keys
    for idx, exp_key in enumerate(exp_keys):
        exp = bioen_data[np.max(theta_series)]['exp']['generic'][exp_key]
        exp_err = bioen_data[np.max(theta_series)]['exp_err']['generic'][exp_key]
        if idx == 0:
            ax.errorbar(idx, exp, yerr=exp_err, fmt='o', color='black',
                        label='Exp. + Error', zorder=2)
        else:
            ax.errorbar(idx, exp, yerr=exp_err, fmt='o', color='black', zorder=2)

        a = np.linspace(0.1, 0.7, num=len(theta_series))
        for i, theta in enumerate(theta_series):
            sim = bioen_data[theta]['sim_wopt']['generic'][exp_key]
            if idx == 0:
                ax.plot(idx, sim, marker='^', color='red', alpha=a[i],
                        label=r"BioEn ($\theta={}$)".format(theta), zorder=3)
            else:
                ax.plot(idx, sim, marker='^', color='red', alpha=a[i], zorder=3)

    ax.set_xticks(range(0, len(exp_keys)))
    ax.set_xlim(-0.5, len(exp_keys)-0.5)
    ax.set_xticklabels(exp_keys)
    ax.set_ylabel(r'Experimental Unit')
    ax.legend(loc=1, ncol=2)
    plt.grid()
    plt.tight_layout()
    return fig



def plot_cd(self, theta_series):
    fig = plt.figure(figsize=[9, 6])
    ax = fig.add_subplot(111)

    exp = self.bioen_data[np.max(theta_series)]['exp']['cd']
    ax.plot(exp[:, 0], exp[:, 1], color='black', linewidth=2.5, label='Exp.', zorder=2)

    a = np.linspace(0.1, 0.7, num=len(theta_series))
    for i, theta in enumerate(theta_series):
        sim = self.bioen_data[theta]['sim_wopt']['cd']
        ax.plot(exp[:, 0], sim, color='red', alpha=a[i], linewidth=3.0,
                label=r"BioEn ($\theta={}$)".format(theta), zorder=3)

    ax.set_xlabel(r'Wavelength')
    ax.set_ylabel(r'Molecular elipticity')
    ax.legend(loc=1, ncol=2)
    plt.grid()
    plt.tight_layout()
    return [fig]

