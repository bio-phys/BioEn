#!/bin/bash

nmodels=$1

path_exp="files/experimental_data"
path_input="files/output_preparation"
path_output="files/output_bioen"

bioen \
    --optimization_minimizer GSL \
    --number_of_models ${nmodels} \
    --models_list ${path_input}/models_scattering.dat \
    --experiments scattering \
    --theta theta.dat \
    --number_of_iterations 10 \
    --scattering_input_pkl ${path_input}/input-bioen-scattering.pkl \
    --scattering_coefficient initial-optimization \
    --output_pkl ${path_output}/bioen-scattering.pkl 



