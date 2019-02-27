
import os
import sys
if sys.version_info >= (3,):
    import pickle
else:
    import cPickle as pickle
import numpy as np

import h5py


######################################################
def get_list_from_dict (new_mydict, *args):
    mylist = []
    for arg in args:
        mylist.append(new_mydict[arg])
    return mylist




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
        #print ('File type is hdf5 file')
        return file_obj

    elif (isinstance(file_obj, str)):
        #print ('File_obj is string')

        extension = os.path.splitext(file_obj)[1]
        basename = os.path.splitext(file_obj)[0]
        fm_hdf5= basename + ".h5"


        hdf5_obj = h5py.File(fm_hdf5, mode)


    else:
        print ('Type ', type(file_obj), ' not valid')
        raise 'File Type not recognized'


    return hdf5_obj




######################################################
def dump(py_obj, file_obj, dataset):

    # it can be a ifile object, string or hdf5 object
    hdf5_obj = openHDF5(file_obj, mode='w')

    file_obj.create_dataset(dataset, data=py_obj)

    hdf5_obj.close()

    return

######################################################
def load (file_obj, dataset):

    hdf5_obj = openHDF5(file_obj, mode='r')

    return file_obj.get(dataset).value
     
######################################################
def dump_by_kw(file_obj, *args, **kwargs):
   
    hdf5_obj = openHDF5(file_obj, mode='w')

    for key, value in kwargs.iteritems():
        hdf5_obj.create_dataset(key, data=value)

######################################################
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


### SHOW KEYS IN HDF5
######################################################
def show_keys(hdf5_obj):
    print "Show keys -> Datasets in file ", hdf5_obj.filename
    for val in hdf5_obj.keys():
        print("--> " + val)


#### WRITE DICTIONARY
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
            #result_dict[key] = value.value
            result_dict[key] = value.value
        if (isinstance(value, h5py.Group)):
            result_dict[key] = load_rec_dict(file_obj, value)

    return result_dict


