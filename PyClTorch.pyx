import cython
cimport cython

cimport cpython.array
import array

import PyTorch

#PyTorch.foo = PyTorch.FloatTensor
#PyTorch.FloatTensor = None

#def class FloatTensor(PyTorch.FloatTensor):
#    def __cinit__(self):
#        print('__cinit__')

#    def __dealloc__(self):
#        print('__dealloc__')

import floattensor_patch

