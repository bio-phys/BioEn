"""Convert pickle input files for the forces tests into HDF5,
read them back in and compare.  Use temporary HDF5 files.
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
tol_grad = 5.e-12

filenames_forces = [
    "./data/data_deer_test_forces_M808xN10.pkl",
    "./data/data_forces_M64xN64.pkl"
]


def test_fileio_forces():
    if sys.version_info >= (3,):
        return
    for filename_pkl in filenames_forces:
        hdf5_file = tempfile.NamedTemporaryFile(mode='w', suffix=".h5", delete=False)
        hdf5_file.close()
        print (filename_pkl, hdf5_file.name)

        # perform file conversion
        keys = ["forces_init", "w0", "y", "yTilde", "YTilde", "theta"]
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
