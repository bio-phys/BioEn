#!/bin/bash

label_pair_id=$1
nmodels=$2

path_input="files/output_preparation"
path_output="files/output_bioen"
path_exp="files/experimental_data"

bioen \
    --optimization_minimizer GSL \
    --experiments deer \
    --number_of_models ${nmodels} \
    --models_list ${path_input}/models_${label_pair_id}.dat \
    --theta theta.dat \
    --number_of_iterations 15 \
    --deer_exp_path ${path_exp} \
    --deer_exp_prefix exp \
    --deer_exp_suffix signal-deer \
    --deer_noise ${path_exp}/exp-error.dat \
    --deer_labels ${label_pair_id} \
    --deer_modulation_depth initial-optimization \
    --deer_input_sim_pkl ${path_input}/data_${label_pair_id}.pkl \
    --output_pkl ${path_output}/bioen_${label_pair_id}.pkl \



