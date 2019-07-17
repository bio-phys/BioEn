#!/bin/bash


path_exp="files/experimental_data"
path_sim="files/simulated_data"
path_output="files/output_bioen"

bioen \
    --optimization_minimizer gsl \
    --optimization_parameters gsl:step_size=0.01,gsl:tol=0.001,gsl:max_iterations=4000 \
    --experiments cd \
    --number_of_models 100 \
    --models_list ${path_sim}/models-cd.dat \
    --theta thetas.dat \
    --cd_sim_path ${path_sim} \
    --cd_sim_prefix conf \
    --cd_sim_suffix cd \
    --cd_exp_path ${path_exp} \
    --cd_exp_prefix exp \
    --cd_exp_suffix cd \
    --cd_noise ${path_exp}/exp-cd-error.dat \
    --output_pkl ${path_output}/bioen_results_cd.pkl \
    --output_weights ${path_output}/bioen_results_weights.dat \
    --output_models_weights ${path_output}/bioen_results_models_weights.dat

###  You can adapt the parameters for GSL as follows, overriding the default parameter file <bioen_optimize.yaml>, for example:
#    --optimization_minimizer gsl \
#    --optimization_parameters gsl:step_size=0.01,gsl:tol=0.001,gsl:max_iterations=2000 \
#    --optimization_verbose \
