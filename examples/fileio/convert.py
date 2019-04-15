#!/usr/bin/env python
"""
Very simple Pickle-to-HDF5 converter.

Usage: convert.py file.pkl file.h5
"""

import sys
import bioen.fileio as fio

fio.convert_to_hdf5(sys.argv[1], sys.argv[2])

