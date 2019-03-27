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
    "./data/data_potra_part_2_logw_M205xN10.pkl",
    "./data/data_16x15.pkl",                       # synthetic test case
    "./data/data_deer_test_logw_M808xN10.pkl",
    "./data/data_potra_part_2_logw_M205xN10.pkl",
    "./data/data_potra_part_1_logw_M808xN80.pkl",  # (*)(1)
    "./data/data_potra_part_2_logw_M808xN10.pkl"
]


def test_fileio_logw():
    if sys.version_info >= (3,):
        return
    for filename_pkl in filenames_logw:
        hdf5_file = tempfile.NamedTemporaryFile(mode='w', suffix=".h5", delete=False)
        hdf5_file.close()
        print (filename_pkl, hdf5_file.name)

        keys = ["GInit", "G", "y", "yTilde", "YTilde", "w0", "theta"]
        fio.convert_to_hdf5(filename_pkl, hdf5_file.name, keys)

        # load pickle
        with open(filename_pkl, 'rb') as ifile:
            x = pickle.load(ifile)

        # load hdf5
        y = fio.load(hdf5_file.name, hdf5_keys=keys)
        os.unlink(hdf5_file.name)

        # compare
        assert(len(x) == len(y))
        for i, elem_x in enumerate(x):
            elem_y = y[i]
            assert(np.array_equal(elem_x, elem_y))

