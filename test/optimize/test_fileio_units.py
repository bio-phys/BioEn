"""Unit tests for the fileio module.
"""

import os
import pytest
import tempfile
from bioen import fileio as fio


@pytest.fixture
def temp_hdf5_file_name():
    hdf5_file = tempfile.NamedTemporaryFile(mode='w', suffix=".h5", delete=False)
    hdf5_file.close()
    return hdf5_file.name


def test_dump_load_dict(temp_hdf5_file_name):
    filename = temp_hdf5_file_name

    mynested = {"var1": "a_string", "var2": 32, "var3": [2, 3, 4]}
    mydict = {"label": "value", "nested": mynested}
    fio.dump(filename, mydict)

    mydict = fio.load(filename, hdf5_deep_mode=True)
# TODO: take care of string decoding
#    assert(mydict["label"] == "value")
#    assert(mydict["nested"]["var1"] == "a_string")
#    assert(mydict["nested"]["var2"] == 32)
#    assert(mydict["nested"]["var3"][0] == 2)
#    assert(mydict["nested"]["var3"][1] == 3)
#    assert(mydict["nested"]["var3"][2] == 4)
    os.remove(filename)


def test_dump_load_list_labeled(temp_hdf5_file_name):
    filename = temp_hdf5_file_name

    data = [1, 2, 3, 5.0]
    keys = ["one", "two", "three", "five.zero"]
    fio.dump(filename, data, hdf5_keys=keys)
    check = fio.load(filename, hdf5_keys=keys)
    for i, item in enumerate(check):
        assert(item == data[i])
    os.remove(filename)


def test_dump_load_list_unlabeled(temp_hdf5_file_name):
    filename = temp_hdf5_file_name

    data = [1, 2, 3, 5.0]
    fio.dump(filename, data)
    check = fio.load(filename)
    for i, item in enumerate(check):
        assert(item == data[i])
    os.remove(filename)
