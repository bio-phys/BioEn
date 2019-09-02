#!/bin/bash

path="data/"

bioen \
    --experiments scattering \
    --number_of_models 5 \
    --models_list ${path}/models-saxs.dat \
    --initial_weights uniform \
    --theta thetas.dat \
    --number_of_iterations 10 \
    --scattering_sim_path ${path} \
    --scattering_sim_prefix lyz \
    --scattering_sim_suffix sim-saxs \
    --scattering_exp_path ${path} \
    --scattering_exp_prefix lyz \
    --scattering_exp_suffix exp \
    --scattering_noise exp-file \
    --scattering_scaling_factor initial-optimization \
    --output_pkl out-scattering.pkl 
    #--optimizationDebug \

bioen \
    --simple_plot \
    --simple_plot_input out-scattering.pkl \
    --simple_plot_output out-scattering.pdf 
