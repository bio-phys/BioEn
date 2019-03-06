
import os, sys
import numpy as np
import pickle
from bioen import fileio as fio


filenames = [
    "./data/data_deer_test_forces_M808xN10.pkl",
    "./data/data_forces_M64xN64.pkl",
    "./data/data_potra_part_2_logw_M205xN10.pkl",  # realistic test case provided by Katrin, has small theta
    "./data/data_16x15.pkl",                       # synthetic test case
    "./data/data_deer_test_logw_M808xN10.pkl",
    "./data/data_potra_part_2_logw_M205xN10.pkl",
    "./data/data_potra_part_1_logw_M808xN80.pkl",   ## (*)(1)
    "./data/data_potra_part_2_logw_M808xN10.pkl"  # ## (*)(2) with default values (tol,step_size) gsl/conj_pr gives NaN
]

 
        
### This code's purpose is to bulk refereces pickle content into .ref.h5 files
for file_name in filenames:
 
    if (os.path.isfile(file_name)):
        ref_file_name = os.path.splitext(file_name)[0] + ".ref"
        with open(ref_file_name, "rb") as f:
            fmin_reference = pickle.load(f)

        h5_file_name= ref_file_name+".h5"
        print (ref_file_name, fmin_reference, " ---> " , h5_file_name)

        fio.dump(h5_file_name, fmin_reference, "reference")

        fmin_ref_load = fio.load(h5_file_name, "reference" )
        print (" Returned value is " , fmin_ref_load)
    else:
        print ("Filename", file_name, " not created" )



