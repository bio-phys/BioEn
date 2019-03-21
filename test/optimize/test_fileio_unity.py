from bioen import fileio as fio
import os


def test_get_dict_from_list():

    mydict = fio.get_dict_from_list(var1='a_string', var2=32, var3=[2,3,4])

    assert(mydict['var1']=='a_string')
    assert(mydict['var2']==32)
    assert(mydict['var3']==[2,3,4])


def test_get_list_from_dict ():

    mydict = { "var1":"a_string" , "var2":32 , "var3":[2,3,4] }

    mylist = fio.get_list_from_dict(mydict, "var1", "var3", "var2")
  
    assert(mylist == ['a_string', [2,3,4], 32] )
    assert(mylist[0]== 'a_string')
    assert(mylist[2]== 32)
    assert(mylist[1][1]== 3)
    assert(True)



def test_dump():

    filename = 'file_test_dump.h5'
    mynested = { "var1":"a_string" , "var2":32 , "var3":[2,3,4] }
    mydict = { "label":"value" , "nested":mynested }
    fio.dump(filename, mydict)
    
    assert(os.path.isfile(filename))

def test_load ():
    filename = 'file_test_dump.h5'
    
    mynested = { "var1":"a_string" , "var2":32 , "var3":[2,3,4] }
    mydict   = { "label":"value"   , "nested":mynested }
    fio.dump(filename, mydict)


    mydict = fio.load(filename)

    assert(True)
    assert(mydict["label"]=='value')
    assert(mydict["nested"]["var1"]=="a_string")
    assert(mydict["nested"]["var2"]==32)
    assert(mydict["nested"]["var3"][0]==2)
    assert(mydict["nested"]["var3"][1]==3)
    assert(mydict["nested"]["var3"][2]==4)

    os.remove(filename)


#def test_convert_to_hdf5():
    # using an integration test for this functions





