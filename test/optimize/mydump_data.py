
import os, sys
import numpy as np
import pickle
from bioen import fileio as fio


filenames_forces = [
    "./data/data_deer_test_forces_M808xN10.pkl",
    "./data/data_forces_M64xN64.pkl"
]

filenames_logw = [
    "./data/data_potra_part_2_logw_M205xN10.pkl",  # realistic test case provided by Katrin, has small theta
    "./data/data_16x15.pkl",                       # synthetic test case
    "./data/data_deer_test_logw_M808xN10.pkl",
    "./data/data_potra_part_2_logw_M205xN10.pkl",
    "./data/data_potra_part_1_logw_M808xN80.pkl",   ## (*)(1)
    "./data/data_potra_part_2_logw_M808xN10.pkl"  # ## (*)(2) with default values (tol,step_size) gsl/conj_pr gives NaN
]


#for filename in filenames_forces:
#    print (filename)
#    with open(filename, 'r') as ifile:
#                [GInit, G, y, yTilde, YTilde, w0, theta] = pickle.load(ifile)
#    fio.dump_by_kw(filename, GInit=GInit, G=G, y=y, yTilde=yTilde, YTilde=YTilde, w0=w0, theta=theta)


for filename in filenames_logw:
    fio.show_keys(filename)

#for filename in filenames_forces:
#    print (filename)
#    with open(filename, 'rb') as ifile:
#        [forces_init, w0, y, yTilde, YTilde, theta] = pickle.load(ifile)
#    #fio.show_keys(filename)
#    fio.dump_by_kw(filename, forces_init=forces_init, w0=w0, y=y, yTilde=yTilde, YTilde=YTilde, theta=theta)    

 
for filename in filenames_forces:
    fio.show_keys(filename)
        

