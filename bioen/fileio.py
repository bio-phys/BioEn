
import os
import sys
if sys.version_info >= (3,):
    import pickle
else:
    import cPickle as pickle
import numpy as np

import h5py


#########################################################
def get_dict_from_list(**kwargs):
    """
    Utility to return a dictionary from a list of pairs of key and value

    Parameters
    ---------
    kwargs: keys and values

    Returns
    -------
    mydict: dictionary
    """
    mydict = {}
    for key, value in kwargs.iteritems():
        mydict[key] = value

    return mydict

#########################################################
def get_list_from_dict (mydict, *args):
    """
    Utility to return a list from a dictionary

    Parameters
    ---------
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


#########################################################
def load_rec_dict(file_obj,group):
    """
    Recursive function to extract elements from a h5 group

    Parameters
    ---------
    file_obj: h5py object handler
    group:  h5group

    Returns
    -------
    mydict: result dictionary
    """

    mydict = {}

    for key,value in group.iteritems():

        if (isinstance(value, h5py.Dataset)):
            mydict = value.value
        if (isinstance(value, h5py.Group)):
            mydict[key] = load_rec_dict(file_obj, value)

    return mydict


#########################################################
### works like load_dict
def load (filename):
    """
    Loads data from a pickle file or an hdf5 file

    Parameters
    ---------
    filename: string, file name

    Returns
    -------
    result: a list for pickle, a dictionary for hdf5
    """

    if (not isinstance(filename, str)):
        raise ("Error; filename must be a string" )


    result = {}

    extension = os.path.splitext(filename)[1]

    if extension == ".pkl":
        with open(filename, 'rb') as ifile:
            result = pickle.load(ifile)

    elif extension == ".h5":
        with h5py.File(filename, "r" ) as hdf5_obj:
            for key,value in hdf5_obj.iteritems():
                if (isinstance(value, h5py.Dataset)):
                    result[key] = value.value
                if (isinstance(value, h5py.Group)):
                    result[key] = load_rec_dict(file_obj, value)
    else:
        raise ("Error; filename extension not recognized (only '.h5' or '.pkl'" )

    return result



#########################################################
    """
    Stores data into a pickle file or an hdf5 file

    Parameters
    ---------
    file_obj: h5py object handler
    key: string, key
    value:
    group:  h5group

    Returns
    -------

    """
def dump_rec_dict(file_obj,key, value, group):

    if (isinstance(value, dict) ):
        for lockey, locvalue in value.iteritems():
            locgroup = group.create_group(lockey)
            dump_rec_dict(file_obj,lockey,locvalue,locgroup)

    else:
        group.create_dataset(key, data=value)

    return

#########################################################
def dump(filename, data):
    """
    Stores data into a pickle file or an hdf5 file

    Parameters
    ---------
    filename: string, file name
    data: a list for pickle and a dictonary for hdf5

    Returns
    -------
    result: a list for pickle, a dictionary for hdf5
    """

    if (not isinstance(filename, str)):
        raise ("Error; filename must be a string" )

    extension = os.path.splitext(filename)[1]
    # if filename is pkl use pickle
    if extension == ".pkl":
        with open(filename, 'wb') as ifile:
            pickle.dump(data, ifile)


    # if filename is h5 use h5py
    elif extension == ".h5":

        with h5py.File(filename, "w" ) as hdf5_obj:

            for key, value in data.iteritems():

                if (isinstance(value, dict) ):
                    mygroup = hdf5_obj.create_group(key)
                    dump_rec_dict(file_obj,key, value, mygroup)
                else:
                    hdf5_obj.create_dataset(key, data=value)

    else:
        raise ("Error; filename extension not recognized (only '.h5' or '.pkl'" )

    return


#########################################################
def convert_to_hdf5(filename_pickle, filename_h5, *args):
    """
    Converts pickle file to hdf5

    Parameters
    ---------
    filename_pickle: string, file name
    filename_h5: string, file name
    args: tags/keys of pickle content

    Returns
    -------
    """

    mydict = {}
    mylist = []

    ## tuple to list
    for arg in args:
        if (isinstance(arg, list)):
            for ind in arg:
                mylist.append(ind)
        else:
            mylist.append(arg)

    ## load pickle content
    with open(filename_pickle, 'rb') as ifile:
        x = pickle.load(ifile)

    ## check if #elem match
    if (len(mylist) != len(x)):
        raise ("List of arguments and pickle content does not match")

    mydict = {}
    ## create dictionary
    for i in range(len(mylist)):
        mydict[mylist[i]] = x[i]

    ## store dictionary
    dump(filename_h5,mydict)

    return
