========================================
 BioEn - Bayesian Inference Of ENsembles
========================================

:Authors:       Katrin Reichel, Jürgen Köfinger, Klaus Reuter, César Allande, Lukas S. Stelzl
:Year:          2018
:Licence:       GPLv3
:Copyright:     © 2018 Katrin Reichel, Jürgen Köfinger, Klaus Reuter, César Allande, Lukas S. Stelzl, Gerhard Hummer
:Citation:      | Reichel K., Stelzl L.S., Köfinger J., Hummer G., J. Phys. Chem. Lett. (2018). `doi:10.1021/acs.jpclett.8b02439<https://doi.org/10.1021/acs.jpclett.8b02439>`_.
                | Hummer G. and Köfinger J., J. Chem. Phys. (2015). `doi:10.1063/1.4937786<https://doi.org/10.1063/1.4937786>`_.


Description
===========

BioEn integrates a broad range of experimental data to refine ensembles of structures.

.. image::  /img/bioen.png

For a detailed description of the procedures and the algorithm, we refer to [Hummer2015,Reichel2018].


BioEn spin-label rotamer refinement
-----------------------------------

Link to **Precision DEER Distances from Spin-Label Ensemble Refinement**

https://github.com/bio-phys/BioEn/tree/master/examples/DEER/rotamer-refinement/POTRA


Dependencies and Software Requirements
======================================

* Python 2.7
* Python packages: sys, numpy, scipy, MDAnalysis, pandas
* GSL (>= 2.1)
* GCC (>= 4.9)

To run Jupyter notebooks ``*.ipynb`` in ./examples/ you need additionally

* Jupyter (https://jupyter.org/)
* Python packages: matplotlib


Installation
============

Installation
---------------------
After loading the dependencies, install the package with::

	BIOEN_OPENMP=1 python setup.py install

BIOEN_OPENMP set to 1 enables OPENMP. On OSX use BIOEN_OPENMP=0.

You can use the --user flag for a local installation, which does not require admin/sudo rights. When you install BioEn locally, please check whether $HOME/.local/bin is on your path. You can add the folder to the path, e.g., by adding export PATH=$HOME/.local/bin:$PATH to your .bashrc file. In a conda-environment install the package without the --user flag.

You may have to set the path to your GSL installation, before running the setup command.

Installation on Linux
---------------------

If you have sudo rights you can install GSL via a package manager such as apt-get.

Installing to GSL if you do not have sudo rights is also simple. Download and unpack GSL and cd to folder with unpacked files.::

        configure  --prefix="~/gsl_install"
        make


If you have installed GSL into a non-standard location, you have to set the GSL path.::

       export GSL_HOME=~/gsl_install

Installation on Mac
-------------------
On a Mac, GSL might have been installed via homebrew::

        brew install gsl

You can find the path to your GSL installation typing::

        brew info gsl


into a terminal and then set the path::

        export GSL_HOME=/usr/local/Cellar/gsl/1.16


Usage
=====

We want to integrate a diverse set of experimental data with simulated observables. Therefore, we implemented three types of chi2-square calculations to use different kinds of experimental data:

* Generic data (chi-square calculation without nuisance parameters)
* DEER/PELDOR data (includes the modulation depth as a nuisance parameter)
* Scattering data (includes the coefficient as a nuisance parameter)


(1) Generic data
----------------
The term generic data refers to experimental data, where measurements provide single data points including noise (e.g. NOE, PREs, chemical shifts, J-couplings, distances, chemical cross-links etc). To use generic data, the bioen options should contain ``--experiments generic``. In the experimental data file (e.g. ``./test/generic/data/exp-generic.dat``), the ID (first column) of a data point (second column) and its noise (third column) has to be provided. The ID refers than to the file from the simulated data (e.g. ``./test/generic/data/sim-noe_1-generic.dat``), in which each line is the simulated data point from a single ensemble member (e.g., simualted data extracted from a trajectory of a MD simulation).

The full list of options for generic data is:

.. code-block:: bash

	--sim_path
	--sim_prefix
	--sim_suffix
	--exp_path
	--exp_prefix
	--exp_suffix
	--data_IDs
	--data_weight
	--input_pkl
	--output_pkl

Please take note of the options ``--sim_path``, ``--sim_prefix``, ``--sim_sufffix``, ``--exp_path``, ``--exp_prefix``, and ``--exp_suffix``. These are useful to define the path to and names of the files. Defaults are provided.


(2) Experimental data from DEER/PELDOR measurements
---------------------------------------------------
For the reweighting with experimental data including a nuisance parameter (here: modulation depth), the structure of the input files is extended and more information is needed. To use DEER data, the bioen options should contain ``--experiments deer``. In the case of DEER data, we can either perform reweighting over an ensemble of conformations with averaged spin-label rotamer states or over an ensemble of spin-label rotamer states with a single protein conformation.

If an ensemble of conformations is investigated, provide for each label pair (e.g. 319-259) a single file of the experimental data (e.g., ``./test/deer/data/exp-319-259-deer.dat``) and ensemble member (e.g., ``./test/deer/data/conf0-319-259-deer.dat``). The experimental data file contains:

.. code-block:: bash

	#time   #raw        #polyfit
	0.0     0.9886054   1.0
	0.008   0.97737117  0.99091340848
	0.016   1.0         0.988879614369
	0.024   0.97842962  0.984631477624
	0.032   0.98185696  0.983339482409

The simulated data file (e.g. ``conf0-319-259-deer.dat``) contains:

.. code-block:: bash

	#time   #simulated_data
	0.0     1.0
	0.008   0.99984697806
	0.016   0.999388027044
	0.024   0.998623491217
	0.032   0.997553943855

Using DEER data in BioEn, the models file (``models-deer.dat``) is of particular interest: listed numbers (model IDs) in this file have to be the same as the deer file names (``conf0-319-259-deer.dat, conf1-319-259-deer.dat, conf2-319-259-deer.dat`` and so on).

If an ensemble of spin-label rotamer states is investigated, we recommend to use the Jupyter notebook ``deer_spin_label_reweighting.ipynb`` in ``./examples/DEER/rotamer-refinement/single_trace/``. Here, the user can define the protein structure and a own rotamer library (or use the default). By executing the cells in the notebook, data preparation, BioEn run, and analysis can be performed in a smooth procedure. The analysis of the BioEn data include also the L-curve analysis. More details on the method are provided in \[Reichel2018].

For both cases, refinement over an ensemble of protein conformations or over spin-label rotamer states, the modulation depth as the nuisance parameter is relevant. With the option ``--deer_modulation_depth``, an initial guess ("<path_to_file>/modulation-depth.dat") can be provided or an initial optimization ("initial-optimization") can be performed for each spin-label pair. As indicated above, the modulation depth is needed to calculate the consistency of the simulated data with the experimental data correctly. To achieve this, we have to iteratively optimize the weights of the ensemble members and the modulation depth. For all cases tested with DEER data, 10 iterations seems to be sufficient until the optimization converges. To do so, we recommend to set the option ``--number_of_iterations`` to **10** or higher.

The full list of options for DEER data is:

.. code-block:: bash

	--deer_sim_path
	--deer_sim_prefix
	--deer_sim_suffix
	--deer_exp_path
	--deer_exp_prefix
	--deer_exp_suffix
	--deer_labels
	--deer_noise
	--deer_modulation_depth
	--deer_input_pkl
	--deer_input_hd5
	--deer_output_pkl
	--deer_input_sim_pkl
	--deer_input_sim_hd5

Please take note of the options ``--deer-sim_path``, ``--deer_sim_prefix``, ``--deer_sim_suffix``, ``--deer_exp_path``, ``--deer_exp_prefix``, and ``--deer_exp_suffix``. These options are useful to define the names of the simulated and experimental files. In addition, please define the spin-label pairs with ``--deer_labels`` (e.g.; "319-259,370-259"), which is also part of the experimental and simulated data file names (see above).


(3) Experimental data from SAXS/WAXS measurements
-------------------------------------------------
BioEn can be used with scattering data like SAXS or WAXS, for which we provide also the optimization of the nuisance parameter (here: coefficient). To use scattering data, the bioen options should contain ``--experiments scattering``. The input data is handled in a similar way as the DEER data, but just for a single scattering curve and not different label-pairs. The standard file format for experimental data (e.g. ``lyz-exp.dat``) is:

.. code-block:: bash

    #   q                 I(q)      error/noise
    4.138455E-02        5.904029    1.555333E-01
    4.371607E-02        5.652469    1.527037E-01
    4.604759E-02        5.533381    1.521723E-01
    4.837912E-02        5.547052    1.474577E-01
    5.071064E-02        5.296281    1.436712E-01


The simulated data file (e.g. ``lyz0-sim-saxs.dat``) contains:

.. code-block:: bash

	#   q               I(q)
	4.138454e-02 	2.906550e+06
	4.371607e-02 	2.865970e+06
	4.604758e-02 	2.823741e+06
	4.837911e-02 	2.779957e+06
	5.071064e-02 	2.734716e+06

To handle different data input, we recommend to use the ipython notebook ``./examples/scattering/scattering_reweighting.ipynb``.

The full list of options for scattering data is:

.. code-block:: bash

 	--scattering_sim_path
	--scattering_sim_prefix.
	--scattering_sim_suffix
	--scattering_exp_pPath
	--scattering_exp_prefix
	--scattering_exp_suffix
	--scattering_noise
	--scattering_coefficient
	--scattering_data_weight
	--scattering_input_pkl
	--scattering_input_hd5
	--scattering_input_sim_pkl
	--scattering_input_sim_hd5
	--scattering_output_pkl


Please take note of the options ``--scattering_sim_prefix``, ``--scattering_sim_sufffix``, ``--scattering_exp_prefix``, and ``--scattering_exp_suffix``. These options are useful to define the names of the files of experimental and simulated.

As indicated above, a nuisance parameter (here: coefficient) is needed to calculate the consistency of the simulated data with the experimental data correctly. To achieve this, we have to iteratively optimize the weights of the ensemble members and the coefficient. For all cases tested with scattering data, 10 iterations seems to be sufficient until the optimization converges. To do so, we recommend to set the option ``--number_of_iterations`` to **10** or higher.


Other options and settings
--------------------------
The initial and reference weights can be set with ``--reference_weights`` and ``--initial_weights``. For both options, one can either choose **uniform** (uniformly distributed weights; default), **random** (randomly distributed weights), or provide a file as input.

As described in [Hummer2015], we have to balance the consistency with the experimental data (chi-square) with the changes in the weights (relative entropy) by the **confidence parameter theta**. We can achieve this aim by the maximum-entropy principle and as such avoid over-fitting. To decide for the correct confidence parameter theta for a specific set of data, usually a theta-series is applied. This means, that for each theta an independent ensemble refinement run is performed. Subsequent L-curve analysis (relative entropy vs. chi-square) leads us to the optimal weight distribution. Please note, that the choice of the confidence parameter depends on the system and data. In the BioEn software package, one can choose ``--theta`` by defining a single value (e.g., 10.0) or a theta-series, which can be provided as a list (e.g., 100.0,10.0,1.0) or a list in a file (e.g., <path_to_file>/thetas.dat).

To check the BioEn results quickly, a simple plot can be generated, that compares experimental data and ensemble averaged simulated data for the used confidence values. Therefore, the following three options have to be set: ``--simple_plot``, ``--simple_plot_input`` and ``--simple_plot_output``. The file name of the output pkl file has to be provided for ``--simple_plot_input``. The data in this pkl file is visualized and saved in a pdf file, which can be specified with ``--simple_plot_output``.


Misc options
------------
The option ``--output_pkl_input_data`` can be used to generate a pkl file of all settings, parameters and weights from the previous BioEn run. This file can then be used afterwards with ``--input_pkl`` to restart the BioEn calculation.


Minimal example
---------------
The minimal amount of input parameters are:

* number of ensemble members (``--number_of_models``)
* list of models (``--models_list``)
* type of experiments (``--experiments``)
* input experimental and simulated data

In case you have data from NMR measurements (e.g. NOEs), a typical invocation would look like this:

.. code-block:: bash

    bioen \
        --number_of_models 10 \
        --models_list <path-to-data>/models-generic.dat \
        --experiments generic \
        --theta 0.01 \
        --sim_path <path-to-data> \
        --exp_path <path-to-data> \
        --data_ids all

We provide example test scripts ``run_bioen*.sh`` in ``./test/generic/``, ``./test/deer/``, and ``./test/scattering/`` to run BioEn with the three mentioned types of data.


Default settings
----------------
The default setting for reweighting is log-weights for the procedure and bfgs2 for the optimization algorithm.


Output
------
Three BioEn output files are generated by default, for which you can choose the file names or leave it with the default naming.

(1) The most useful BioEn output file is in pickle (pkl) format. Choose the name of this file with the option ``--output_pkl``. The default file name is **bioen_result.pkl**. This pkl file contains all relevant information from the weight optimization including experimental data, ensemble averaged data, (reference, initial, and optimized) weights, consistency of experimental data with experimental data (chi-squared), relative entropy, etc. For a complete analysis of your BioEn calculations, this file is essential.

(2) The second file contains a list of weights in text file format. The name can be choosen with ``--output_weights``. The default name is **bioen_result_weights.dat**. But careful, it generates this file only for the smallest confidence value theta.

(3) The third files contains for each ensemble member the corresponding weight. This file is similar to the second file, however, it includes also the IDs of each ensemble member and is as such in a tabular form. The name of the file can be chosen by ``--output_models_weights`` with the default file name **bioen_result_models_weights.dat**. Also here, this file is generated from the smallest confidence value theta.


Misc information
----------------
We recommend to have a close look at the files in the folders ``./test/generic/``, ``./test/deer/``, and ``./test/scatter/``. These files can be used to understand and transfer the own scientific questions to BioEn. Lines including ``#`` are in general ignored.

For further options and more information, type::

	bioen --help


Help
====

Please, if you have an issue with the software, open an issue here on the github repository. If you have any questions, please contact bioen@biophys.mpg.de.


References
==========

.. Articles
.. --------

.. [Reichel2018] Reichel K., Stelzl Lukas S., Köfinger J., Hummer G., Precision DEER Distances from Spin-Label Reweighting, J. Phys. Chem. Lett., in press (2018). doi:10.1021/acs.jpclett.8b02439_.

.. [Hummer2015] Hummer G. and Koefinger J., Bayesian Ensemble Refinement by Replica Simulations and Reweighting. J. Chem. Phys. 143(24):12B634_1 (2015). doi:10.1063/1.4937786_.

.. _doi:10.1021/acs.jpclett.8b02439: https://doi.org/10.1021/acs.jpclett.8b02439

.. _doi:10.1063/1.4937786: https://doi.org/10.1063/1.4937786
