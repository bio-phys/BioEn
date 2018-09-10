#!/bin/bash

path_output_bioen=$1
nmodels=$2
spin_label_pair=$3
file_id=$4

path_exp="files/experimental_data"
path_input="files/output_preparation"

bioen \
    --experiments deer \
    --number_of_models ${nmodels} \
    --models_list ${path_input}/models_${file_id}.dat \
    --theta theta.dat \
    --number_of_iterations 10 \
    --deer_exp_path ${path_exp} \
    --deer_exp_prefix exp \
    --deer_exp_suffix signal-deer \
    --deer_noise file \
    --deer_labels ${spin_label_pair} \
    --deer_modulation_depth ${path_exp}/models-moddepth_potra.dat \
    --deer_input_sim_pkl ${path_input}/data_${file_id}.pkl \
    --output_pkl ${path_output_bioen}/test-${nmodels}-${file_id}.pkl \



