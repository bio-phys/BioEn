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


def load(filename, hdf5_keys=[]):
    """
    Load data from a pickle file or an HDF5 file, depending on the filename suffix.

    Parameters
    ----------
    filename: string, file name, suffix decides if pickle or HDF5 is read.
    hdf5_keys: optional list of keys (strings) to specify the datasets from the HDF5 file

    Returns
    -------
    result: object restored by pickle, typically a list for legacy BioEN data, or
            a dictionary for hdf5 in case hdf5_keys is empty, or
            a list for hdf5 containing only the datasets for the given hdf5_keys
    """

    extension = os.path.splitext(filename)[1]

    if extension == ".pkl":
        result = load_pickle(filename)
    elif extension == ".h5":
        result = load_hdf5(filename, hdf5_keys)
    else:
        raise ValueError("filename extension not recognized (only '.h5' or '.pkl')")

    return result


def dump(filename, data):
    """
    Stores data into a pickle file or an hdf5 file, depending on the filename suffix.

    Parameters
    ----------
    filename: string, file name, suffix decides if pickle or HDF5 is written.
    data: a list for pickle, or a dictionary with string-keys for hdf5

    Returns
    -------
    """

    extension = os.path.splitext(filename)[1]

    if extension == ".pkl":
        dump_pickle(filename, data)
    elif extension == ".h5":
        dump_hdf5(filename, data)
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
    n = len(x)

    # in case no or wrong labels were given, simply label using alphabetic letters
    if len(hdf5_keys) != n:
        hdf5_keys = []
        for i in range(n):
            hdf5_keys.append(string.ascii_letters[i])

    mydict = {}
    for i in range(n):
        mydict[hdf5_keys[i]] = x[i]

    # store data to HDF5
    dump_hdf5(filename_h5, mydict)


# --- low level functions below ---


def load_pickle(file_name):
    with open(file_name, 'rb') as fp:
        return pickle.load(fp)


def load_hdf5(file_name, hdf5_keys=[]):
    with h5py.File(file_name, "r") as hdf5_obj:
        if hdf5_keys:
            result = []
            for key in hdf5_keys:
                value = hdf5_obj[key]
                if isinstance(value, h5py.Dataset):
                    result.append(value.value)
        else:
            result = {}
            for key, value in sorted(hdf5_obj.items()):
                if isinstance(value, h5py.Dataset):
                    result[key] = value.value
                if isinstance(value, h5py.Group):
                    result[key] = load_rec_dict(value)
    return result


def dump_pickle(file_name, data):
    with open(file_name, 'wb') as fp:
        pickle.dump(data, fp)


def dump_hdf5(file_name, data):
    with h5py.File(file_name, "w") as hdf5_obj:
        for key, value in data.items():
            if isinstance(value, dict):
                mygroup = hdf5_obj.create_group(key)
                dump_rec_dict(hdf5_obj, key, value, mygroup)
            else:
                hdf5_obj.create_dataset(key, data=value)


def get_dict_from_list(**kwargs):
    """
    Return a dictionary from a list of pairs of key and value.

    Parameters
    ----------
    kwargs: keys and values

    Returns
    -------
    mydict: dictionary
    """

    mydict = {}
    for key, value in kwargs.items():
        mydict[key] = value
    return mydict


def get_list_from_dict(mydict, *args):
    """
    Return a list from a dictionary.

    Parameters
    ----------
    mydict: dictionary
    args: keys to extract from the dictionary

    Returns
    -------
    mylist: list
    """

    mylist = []
    for arg in args:
        if (isinstance(arg, list)):
            for ind in arg:
                mylist.append(mydict[ind])
        else:
            mylist.append(mydict[arg])
    return mylist


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
            mydict = value.value
        if (isinstance(value, h5py.Group)):
            mydict[key] = load_rec_dict(value)
    return mydict


def dump_rec_dict(file_obj, key, value, group):
    """
    Store data into a pickle file or an hdf5 file.

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


# def retrieve_name(var):
#     import inspect
#     callers_local_vars = inspect.currentframe().f_back.f_locals.items()
#     return [var_name for var_name, var_val in callers_local_vars if var_val is var]

# def name(pkl_file):
#     pkl = load_pickle(pkl_file)
#     if isinstance(pkl, (list, tuple)):
#         for elem in pkl:
#             n = retrieve_name(elem)
#             print(n)
