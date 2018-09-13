#!/bin/bash

path="data"

bioen \
    --optimization_method log-weights \
    --optimization_algorithm bfgs \
    --optimization_minimizer scipy \
    --experiments deer \
    --number_of_models 10 \
    --models_list ${path}/models-deer.dat \
    --theta thetas.dat \
    --number_of_iterations 10 \
    --deer_sim_path ${path} \
    --deer_sim_prefix conf \
    --deer_sim_suffix deer \
    --deer_exp_path ${path} \
    --deer_exp_prefix exp \
    --deer_exp_suffix deer \
    --deer_labels 319-259 \
    --deer_noise ${path}/exp-error.dat \
    --deer_modulation_depth ${path}/moddepth-deer.dat \
    --output_pkl out-deer.pkl \

bioen \
    --simple_plot \
    --simple_plot_input out-deer.pkl \
    --simple_plot_output out-deer.pdf
