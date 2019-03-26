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


def load(filename, hdf5_keys=[]):
    """
    Load data from a pickle file or an hdf5 file.

    Parameters
    ----------
    filename: string, file name

    Returns
    -------
    result: a list for pickle, a dictionary for hdf5
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
    Stores data into a pickle file or an hdf5 file

    Parameters
    ----------
    filename: string, file name
    data: a list for pickle and a dictionary for hdf5

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


def convert_to_hdf5(filename_pickle, filename_h5, *args):
    """
    Convert pickle file to hdf5.

    Parameters
    ----------
    filename_pickle: string, file name
    filename_h5: string, file name
    args: tags/keys of pickle content

    Returns
    -------
    """

    mydict = {}
    mylist = []

    # tuple to list
    for arg in args:
        if (isinstance(arg, list)):
            for ind in arg:
                mylist.append(ind)
        else:
            mylist.append(arg)

    # load pickle content
    ext = os.path.splitext(filename_pickle)[1]
    if (ext != ".pkl"):
        raise ValueError("pickle file name must have '.pkl' extension")

    x = load_pickle(filename_pickle)

    # check if #elem match
    if (len(mylist) != len(x)):
        raise ValueError("list of arguments (length " + str(len(mylist)) +
                         "), and pickle (length " + str(len(x)) + ") do not match")

    mydict = {}
    # create dictionary
    for i in range(len(mylist)):
        mydict[mylist[i]] = x[i]

    # store dictionary
    ext = os.path.splitext(filename_h5)[1]
    if (ext != ".h5"):
        raise ValueError("hdf5 file name must have '.h5' extension")

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
            for key, value in hdf5_obj.items():
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

    for key, value in group.items():
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
