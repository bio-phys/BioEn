
import os
import sys
if sys.version_info >= (3,):
    import pickle
else:
    import cPickle as pickle
import numpy as np

import h5py

## Utility to list elements within a h5 file
######################################################
def show_keys(file_obj):
    hdf5_obj = openHDF5(file_obj, mode='r')
    print "Show keys -> Datasets in file ", hdf5_obj.filename
    for val in hdf5_obj.keys():
        print("--> " + val)

## Utility to get a list of elements from dictionary
## To obtain the same behavior as in previous pickle mechanism
######################################################
def get_list_from_dict (new_mydict, *args):
    mylist = []
    for arg in args:
        mylist.append(new_mydict[arg])
    return mylist



## General function to open h5py file.
## Accepted:
##      string: filename
##      pickle_obj:
##      h5py_obj:
######################################################
def openHDF5 (file_obj, mode ='rb'):

    if isinstance(file_obj, file):
        # File type pickle
        filename = file_obj.name
        #extension = os.path.splitext(filename)[1]
        basename = os.path.splitext(filename)[0]
        fm_hdf5= basename + ".h5"
        print('Pickle is deprecated, using HDF5 file instead : ' + fm_hdf5 )

        hdf5_obj = h5py.File(fm_hdf5, mode)

    elif (isinstance(file_obj, h5py._hl.files.File)) :
        return file_obj

    elif (isinstance(file_obj, str)):
        extension = os.path.splitext(file_obj)[1]
        basename = os.path.splitext(file_obj)[0]
        fm_hdf5= basename + ".h5"
        hdf5_obj = h5py.File(fm_hdf5, mode)


    else:
        print ('Type ', type(file_obj), ' not valid')
        raise 'File Type not recognized'


    return hdf5_obj

### dump and load serve for storing a single element
######################################################
def dump(file_obj, py_obj,  dataset='data'):

    # it can be a ifile object, string or hdf5 object
    hdf5_obj = openHDF5(file_obj, mode='w')

    hdf5_obj.create_dataset(dataset, data=py_obj)

    hdf5_obj.close()

    return

######################################################
def load (file_obj, dataset='data'):

    hdf5_obj = openHDF5(file_obj, mode='r')

    mykeys = hdf5_obj.keys()

    if ( not dataset in hdf5_obj.keys() ) :
        raise ("Error; data set '" + dataset + "' does not exists")

    return hdf5_obj.get(dataset).value

## dump_by_kw and lload_by_kw, allow to specify several
## data to be saved as arguments
## useful when converintg old pickle files into h5
#####################################################
def dump_by_kw(file_obj, *args, **kwargs):

    hdf5_obj = openHDF5(file_obj, mode='w')

    for key, value in kwargs.iteritems():
        hdf5_obj.create_dataset(key, data=value)

## load limits the loaded content from the file
#####################################################
def load_by_kw(file_obj, *args):

    hdf5_obj = openHDF5(file_obj, mode='r')

    result_dict = {}
    mykeys = hdf5_obj.keys()

    for arg in args:
        if arg in mykeys:
            result_dict[arg] = hdf5_obj[arg].value
        else:
            print "Dataset ", arg , " is not present in this file"


    return result_dict



#### WRITE DICTIONARY
## Should be the standard way to load and store
## data. Always working with dictionaries because
## they have key and value
## It should work for nested dictionaries.
######################################################
def dump_rec_dict(file_obj,key, value, group):

    if (isinstance(value, dict) ):
        for lockey, locvalue in value.iteritems():
            locgroup = group.create_group(lockey)
            dump_rec_dict(file_obj,lockey,locvalue,locgroup)

    else:
        group.create_dataset(key, data=value)


######################################################
def dump_dict(file_obj, mydict):

    hdf5_obj = openHDF5(file_obj, mode='w')

    for key, value in mydict.iteritems():

        if (not isinstance(value, dict) ):
            hdf5_obj.create_dataset(key, data=value)
        else:
            mygroup = hdf5_obj.create_group(key)
            dump_rec_dict(file_obj,key, value, mygroup)



#### LOAD DICTIONARY
######################################################
def load_rec_dict(file_obj,group):

    result_dict = {}

    for key,value in group.iteritems():

        if (isinstance(value, h5py.Dataset)):
            #result_dict[key] = value.value
            result_dict = value.value
        if (isinstance(value, h5py.Group)):
            result_dict[key] = load_rec_dict(file_obj, value)

    return result_dict




######################################################
def load_dict(file_obj):

    hdf5_obj = openHDF5(file_obj, mode='r')

    result_dict = {}

    for key,value in hdf5_obj.iteritems():
        if (isinstance(value, h5py.Dataset)):
            result_dict[key] = value.value
        if (isinstance(value, h5py.Group)):
            result_dict[key] = load_rec_dict(file_obj, value)

    return result_dict


