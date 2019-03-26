
import os
import sys
import numpy as np
import pickle
from bioen import optimize
#from bioen import fileio_modif as fio
from bioen import fileio as fio
import h5py

tol = 5.e-14
tol_grad = 5.e-12

filenames_forces = [
    "./data/data_deer_test_forces_M808xN10.pkl",
    "./data/data_forces_M64xN64.pkl"
]


def test_fileio_forces():

    for filename_pkl in filenames_forces:

        filename_hdf5 = os.path.splitext(filename_pkl)[0] + ".h5"
        print (filename_pkl, filename_hdf5)

        mylist = ["forces_init", "w0", "y", "yTilde", "YTilde", "theta"]
        fio.convert_to_hdf5(filename_pkl, filename_hdf5, mylist)

        # load pickle
        with open(filename_pkl, 'rb') as ifile:
            x = pickle.load(ifile)

        # load hdf5
        mydict = fio.load(filename_hdf5)

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
