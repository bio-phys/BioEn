#!/bin/bash

nmodels=$1

path_exp="files/experimental_data"
path_input="files/output_preparation"
path_output="files/output_bioen"

bioen \
    --optimizationMethod log-weights \
    --optimizationAlgorithm bfgs \
    --optimizationMinimizer scipy \
    --optimizationDebug \
    --numberOfModels ${nmodels} \
    --modelsList ${path_input}/models_scattering.dat \
    --experiments scattering \
    --theta theta.dat \
    --numberOfIterations 10 \
    --scatteringInputPkl ${path_input}/input-bioen-scattering.pkl \
    --scatteringCoefficient initial-optimization \
    --outputPkl ${path_output}/test-${nmodels}-scattering.pkl 



