fileio, TODO: update to newest implementation
=============================================

The Fileio implementation helps to overcome the previous limitation of
the IO pickle format by using hdf5 files.

One of the main differences between pickle and hdf5 is how hdf5
structures the data. In pickle, it is necessary to retrieve the data in
the same order as the data has been saved. In hdf5 the data is
structured similarly as as a dictionary. It is necessary to assign a
label for every data element and therefore it can be accessed by
indexing the label into the structure (similarly as a python
dictionary).

Fileio facilitates the transition from pickle to hdf5 by proving a
unified interface to pickle and hdf5 files, and also provides a
conversion mechanism from pickle to hdf5.

The following example pretends to illustrate how to transition from the
current pickle IO format to hdf5:

Current usage with pickle
-------------------------

Load
~~~~

.. code:: python

        with open('./data/data_forces_M64xN64.pkl', 'rb') as ifile:
            [forces_init, w0, y, yTilde, YTilde, theta] = pickle.load(ifile)

Store
~~~~~

.. code:: python

        with open('./data/data_forces_M64xN64.pkl', "wb") as ifile:
            pickle.dump([forces_init, w0, y, yTilde, YTilde, theta], ifile)

Fileio using pickle format
--------------------------

Load
~~~~

.. code:: python

        from bioen import fileio as fio
        [forces_init, w0, y, yTilde, YTilde, theta] = fio.load('./data/data_forces_M64xN64.pkl')

Store
~~~~~

.. code:: python

        from bioen import fileio as fio
        fio.dump('./data/data_forces_M64xN64.pkl', [forces_init, w0, y, yTilde, YTilde, theta])

Fileio using hdf5 format (returns a dictionary) - (\*\*) and transform the data into a list or from list to dictionary.
-----------------------------------------------------------------------------------------------------------------------

Load
~~~~

.. code:: python

        from bioen import fileio as fio
        mydict = fio.load('data.h5')
        [forces_init, w0, y, yTilde, YTilde, theta] = fio.get_list_from_dict(mydict,"forces_init", "w0", "y", "yTilde", "YTilde", "theta")

or

.. code:: python

        from bioen import fileio as fio
        mydict = fio.load('data.h5')
        mylist = ["forces_init", "w0", "y", "yTilde", "YTilde", "theta"]
        [forces_init, w0, y, yTilde, YTilde, theta] = fio.get_list_from_dict(mydict, mylist)

Store
~~~~~

.. code:: python

        from bioen import fileio as fio
        mydict = fio.get_dict_from_list(forces_init=forces_init, w0=w0, y=y, yTilde=yTilde, YTilde=YTilde, theta=theta)
        fio.dump ('data.h5', mydict)

(\*\*) Optional; This can be useful to transition from the current
list-like format to a dictionary-like

Fileio converting a pickle file into a new hdf5 file
----------------------------------------------------

.. code:: python

        from bioen import fileio as fio
        mylist=["forces_init","w0","y","yTilde","YTilde","theta"]
        fio.convert_to_hdf5('data.pkl','data.h5',mylist)

Recommendations
===============

Three steps transition:

1) Transform the current pickle files into hdf5 files by using the
   conversion tool. This has to be a suppervised conversion because the
   tags/labels for the data must be specified.

   e.g.:

.. code:: python

        from bioen import fileio as fio

        ### Tags for the data
        mylist=["forces_init","w0","y","yTilde","YTilde","theta"]
        ### Convert pickle into a new h5 file
        fio.convert_to_hdf5('data.pkl','data.h5',mylist)

2) Replace the pickle calls for Fileio pickle calls

.. code:: python

        # Previous pickle
        #with open('./data/data_forces_M64xN64.pkl', 'rb') as ifile:
        #    [forces_init, w0, y, yTilde, YTilde, theta] = pickle.load(ifile)

        # Fileio pickle

        from bioen import fileio as fio
        [forces_init, w0, y, yTilde, YTilde, theta] = fio.load('./data/data_forces_M64xN64.pkl')

3) Replace the filename extension from '.pkl' to '.h5' and call the data
   type converter (dict-list/dict-list)

.. code:: python

        from bioen import fileio as fio
        mydict = fio.load('data.h5')
        [GInit, G, y, yTilde, YTilde, w0, theta] = fio.get_list_from_dict(new_mydict,"GInit", "G", "y", "yTilde", "YTilde", "w0", "theta")
