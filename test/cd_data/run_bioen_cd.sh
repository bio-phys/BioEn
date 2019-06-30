#!/bin/bash

path="data"

bioen \
    --optimization_method log-weights \
    --optimization_algorithm bfgs \
    --optimization_minimizer scipy \
    --experiments cd \
    --number_of_models 10 \
    --models_list ${path}/models-cd.dat \
    --theta thetas.dat \
    --cd_sim_path ${path} \
    --cd_sim_prefix conf \
    --cd_sim_suffix cd \
    --cd_exp_path ${path} \
    --cd_exp_prefix exp \
    --cd_exp_suffix cd \
    --cd_noise ${path}/exp-cd-error.dat \
    --output_pkl out-cd.pkl \

bioen \
    --simple_plot \
    --simple_plot_input out-cd.pkl \
    --simple_plot_output out-cd.pdf
