#!/bin/bash


path="data"
flex_ids="noe_1,noe_2,distance_1,pre_1,r2-J1JNC"

bioen \
    --number_of_models 10 \
    --models_list ${path}/models-generic.dat \
    --experiments generic \
    --theta thetas.dat \
    --sim_path ${path} \
    --exp_path ${path} \
    --data_ids ${flex_ids} \
    --output_pkl bioen_result.pkl 


bioen \
    --simple_plot \
    --simple_plot_input bioen_result.pkl \
    --simple_plot_output bioen_result.pdf 
