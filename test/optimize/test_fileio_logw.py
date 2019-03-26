"""Convert pickle input files for the logw tests into hdf5,
read them back in and compare.
"""

import os
import sys
import numpy as np
import pickle
from bioen import optimize
from bioen import fileio as fio
import h5py
import tempfile

tol = 5.e-14
tol_grad = 5.e-14

filenames_logw = [
    # realistic test case provided by Katrin, has small theta
    "./data/data_potra_part_2_logw_M205xN10.pkl",
    "./data/data_16x15.pkl",                       # synthetic test case
    "./data/data_deer_test_logw_M808xN10.pkl",
    "./data/data_potra_part_2_logw_M205xN10.pkl",
    "./data/data_potra_part_1_logw_M808xN80.pkl",  # (*)(1)
    # ## (*)(2) with default values (tol,step_size) gsl/conj_pr gives NaN
    "./data/data_potra_part_2_logw_M808xN10.pkl"
]


def test_fileio_logw():
    if sys.version_info >= (3,):
        return
    for filename_pkl in filenames_logw:
        # filename_hdf5 = os.path.splitext(filename_pkl)[0] + ".h5"
        filename_hdf5 = tempfile.NamedTemporaryFile(mode='w', suffix=".h5", delete=False)
        filename_hdf5.close()
        print (filename_pkl, filename_hdf5.name)

        mylist = ["GInit", "G", "y", "yTilde", "YTilde", "w0", "theta"]
        fio.convert_to_hdf5(filename_pkl, filename_hdf5.name, mylist)

        # load pickle
        with open(filename_pkl, 'rb') as ifile:
            x = pickle.load(ifile)

        # load hdf5
        mydict = fio.load(filename_hdf5.name)
        os.unlink(filename_hdf5.name)

        # compare
        for i in range(len(mylist)):
            if (np.isscalar(x[i])):
                df = optimize.util.compute_relative_difference_for_values(
                    x[i], mydict[mylist[i]])
                print("\t ({}) [{:14s}]\t - relative difference of scalar = {}  --> values {} .. {}".format(
                    i, mylist[i], df, x[i], mydict[mylist[i]]))
                assert(df < tol)
            else:
                dg, idx = optimize.util.compute_relative_difference_for_arrays(
                    x[i], mydict[mylist[i]])
                print(
                    "\t ({}) [{:14s}]\t - relative difference of vector = {} (maximum at index {})".format(i, mylist[i], dg, idx))
                assert(dg < tol_grad)
