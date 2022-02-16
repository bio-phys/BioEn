"""Module wrapping pickle- and h5py-based IO into a common interface.
"""

import os
import sys
import numpy as np
# disable FutureWarning, intended to warn H5PY developers, but may confuse our users
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
if sys.version_info >= (3,):
    import pickle
else:
    import cPickle as pickle
import string


def load(filename, hdf5_deep_mode=False, hdf5_keys=[]):
    """
    Load data from a pickle file or an HDF5 file, depending on the filename suffix.

    Parameters
    ----------
    filename: string, file name, suffix decides if pickle or HDF5 is read.
    hdf5_deep_mode: bool, read complete hdf5 file and return it in a dictionary
    hdf5_keys: optional list of keys (strings) to specify the datasets from the HDF5 file

    Returns
    -------
    result: object restored by pickle, typically a list for legacy BioEN data, or
            a list for hdf5 containing only the datasets for the given hdf5_keys, or
            a dictionary for hdf5 in case hdf5_deep_mode is True
    """

    extension = os.path.splitext(filename)[1]

    if extension == ".pkl":
        result = load_pickle(filename)
    elif extension == ".h5":
        result = load_hdf5(filename, hdf5_deep_mode, hdf5_keys)
    else:
        raise ValueError("filename extension not recognized (only '.h5' or '.pkl')")

    return result


def dump(filename, data, hdf5_keys=[]):
    """
    Stores data into a pickle file or an hdf5 file, depending on the filename suffix.

    Parameters
    ----------
    filename: string, file name, suffix decides if pickle or HDF5 is written.
    data: a list for pickle,
          a a list or a dictionary with string-keys for hdf5
    hdf5_keys: optional, list of keys (group labels) to use when storing a list to HDF5

    Returns
    -------
    """

    extension = os.path.splitext(filename)[1]

    if extension == ".pkl":
        dump_pickle(filename, data)
    elif extension == ".h5":
        dump_hdf5(filename, data, hdf5_keys)
    else:
        raise ValueError("filename extension not recognized (only '.h5' or '.pkl')")


def convert_to_hdf5(filename_pickle, filename_h5, hdf5_keys=[]):
    """
    Convert pickle file to hdf5 file.

    Parameters
    ----------
    filename_pickle: string, file name
        The pickle file is expected to contain a flat list of numpy arrays of scalars,
        as it was used in the early days of BioEN for the file IO
    filename_h5: string, file name
    hdf5_keys: keys to be used to label the datasets when storing to HDF5

    Returns
    -------
    """

    # load pickle content
    x = load_pickle(filename_pickle)
    assert(isinstance(x, (list, tuple)))

    # store data to HDF5
    dump_hdf5(filename_h5, x, hdf5_keys)



# --------- low(er) level functions below ---------



def load_pickle(file_name):
    with open(file_name, 'rb') as fp:
        return pickle.load(fp)


def load_hdf5(file_name, hdf5_deep_mode=False, hdf5_keys=[]):
    with h5py.File(file_name, "r") as hdf5_obj:
        if hdf5_deep_mode:
            # return the whole hdf5 file in a dictionary
            result = {}
            for key, value in sorted(hdf5_obj.items()):
                if isinstance(value, h5py.Dataset):
                    result[key] = value[()]
                elif isinstance(value, h5py.Group):
                    result[key] = load_rec_dict(value)
        else:
            # return top-level content in a list
            if hdf5_keys:
                # return the top-level datasets specified by hdf5_keys in a list
                result = []
                for key in hdf5_keys:
                    value = hdf5_obj[key]
                    if isinstance(value, h5py.Dataset):
                        result.append(value[()])
            else:
                # return all the top-level datasets in a list
                result = []
                for key, value in sorted(hdf5_obj.items()):
                    if isinstance(value, h5py.Dataset):
                        result.append(value[()])
    return result


def dump_pickle(file_name, data):
    with open(file_name, 'wb') as fp:
        pickle.dump(data, fp)


def dump_hdf5(file_name, data, data_labels=[]):
    """Write data to an HDF5 file.

    Parameters
    ----------
    data: list of numpy arrays, dict of numpy arrays with string keys
    data_labels: string labels used to name the list elements
    """

    def _generate_label(i):
        """Generate a string label based on the integer i, following the
        scheme "AA", "AB", ..., "BA", "BB", ..., "ZA", ..., "ZZ".
        """
        n = len(string.ascii_uppercase)
        return "{}{}".format(string.ascii_uppercase[i//n],
                             string.ascii_uppercase[i%n])

    with h5py.File(file_name, "w") as hdf5_obj:
        if isinstance(data, (list, tuple)):
            n = len(data)
            # in case no or wrong labels were given, create artificial sortable labels
            if len(data_labels) != n:
                data_labels = []
                for i in range(n):
                    data_labels.append(_generate_label(i))
            for i in range(n):
                hdf5_obj.create_dataset(data_labels[i], data=data[i])
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    g = hdf5_obj.create_group(key)
                    dump_rec_dict(hdf5_obj, key, value, g)
                else:
                    hdf5_obj.create_dataset(key, data=value)
        else:
            raise TypeError("data type unsupported")


def load_rec_dict(group):
    """
    Recursive function to extract elements from a h5 group.

    Parameters
    ----------
    file_obj: h5py object handler
    group: h5group

    Returns
    -------
    mydict: result dictionary
    """

    mydict = {}

    for key, value in sorted(group.items()):
        if (isinstance(value, h5py.Dataset)):
            mydict = value[()]
        if (isinstance(value, h5py.Group)):
            mydict[key] = load_rec_dict(value)
    return mydict


def dump_rec_dict(file_obj, key, value, group):
    """
    Recursive function to store data into a pickle file or an hdf5 file.

    Parameters
    ----------
    file_obj: h5py object handler
    key: string, key
    value:
    group:  h5group

    Returns
    -------
    """
    if (isinstance(value, dict)):
        for lockey, locvalue in value.items():
            locgroup = group.create_group(lockey)
            dump_rec_dict(file_obj, lockey, locvalue, locgroup)
    else:
        group.create_dataset(key, data=value)
