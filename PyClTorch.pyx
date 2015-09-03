from __future__ import print_function

import cython
cimport cython

cimport cpython.array
import array

import PyTorch
#cimport PyTorch

#PyTorch.foo = PyTorch.FloatTensor
#PyTorch.FloatTensor = None

#def class FloatTensor(PyTorch.FloatTensor):
#    def __cinit__(self):
#        print('__cinit__')

#    def __dealloc__(self):
#        print('__dealloc__')

cdef extern from "LuaHelper.h":
    cdef struct lua_State
    void *getGlobal(lua_State *L, const char *name1, const char *name2);
    void require(lua_State *L, const char *name)

cdef extern from "THClGeneral.h":
    cdef struct THClState

cdef extern from "THTensor.h":
    cdef struct THFloatTensor

cdef extern from "THClTensor.h":
    cdef struct THClTensor
    THClTensor *THClTensor_newWithSize1d(THClState *state, int device, long size0)
    THClTensor *THClTensor_newWithSize2d(THClState *state, int device, long size0, long size1)
    void THClTensor_free(THClState *state, THClTensor *tensor)

cdef extern from "THClTensorCopy.h":
    void THClTensor_copyFloat(THClState *state, THClTensor *self, THFloatTensor *src)

cdef extern from "THClTensorMath.h":
    float THClTensor_sumall(THClState *state, THClTensor *self)

cdef extern from "clnnWrapper.h":
    THClState *getState(lua_State *L)

cimport PyTorch

cdef class ClTensor(object):
    cdef THClTensor *native

    def __cinit__(ClTensor self, *args):
        print('ClTensor.__cinit__')
        if len(args) > 0:
            for arg in args:
                if not isinstance(arg, int):
                    raise Exception('cannot provide arguments to initializer')
            if len(args) == 1:
                self.native = THClTensor_newWithSize1d(clGlobalState.state, 0, args[0])  # FIXME get device from state
            elif len(args) == 2:
                self.native = THClTensor_newWithSize2d(clGlobalState.state, 0, args[0], args[1])  # FIXME get device from state
            else:
                raise Exception('Not implemented, len(args)=' + str(len(args)))

    def __dealloc__(ClTensor self):
        print('ClTensor.__dealloc__')
        THClTensor_free(clGlobalState.state, self.native)

    def copy(ClTensor self, PyTorch._FloatTensor src):
        THClTensor_copyFloat(clGlobalState.state, self.native, src.thFloatTensor)
        return self

    def sum(ClTensor self):
        return THClTensor_sumall(clGlobalState.state, self.native)

def FloatTensorToClTensor(PyTorch._FloatTensor floatTensor):
    cdef PyTorch._FloatTensor size = floatTensor.size()
    cdef ClTensor clTensor
    if floatTensor.dims() == 1:
        clTensor = ClTensor(int(size[0]))
    elif floatTensor.dims() == 2:
        clTensor = ClTensor(int(size[0]), int(size[1]))
    elif floatTensor.dims() == 3:
        clTensor = ClTensor(int(size[0]), int(size[1]), int(size[2]))
    elif floatTensor.dims() == 4:
        clTensor = ClTensor(int(size[0]), int(size[1]), int(size[2]), int(size[3]))
    else:
        raise Exception('not implemented')
    clTensor.copy(floatTensor)
    return clTensor

import floattensor_patch

#cdef extern from "THRandom.h":
#    cdef struct THGenerator

# This should go into a .pxd or similar probably:
#cdef class GlobalState(object):
###    cdef PyTorchState *state
#    cdef lua_State *L
#    cdef THGenerator *generator

#    def __cinit__(GlobalState self):
#        print('GlobalState.__cinit__')
##        self.state = initPyTorchState();

#    def __dealloc__(self):
#        print('GlobalState.__dealloc__')

#    cdef lua_State *getL(self):  # this is mostly a migration path, we will push this downwards, and out of htis layer
#        return getL(self.state)

#cdef class ClGlobalState(object):
##    cdef PyTorchState *state
##    cdef lua_State *L
##    cdef THGenerator *generator

#    def __cinit__(ClGlobalState self):
#        print('ClGlobalState.__cinit__')
##        self.state = initPyTorchState();

#    def __dealloc__(self):
#        print('ClGlobalState.__dealloc__')

#    cdef lua_State *getL(self):  # this is mostly a migration path, we will push this downwards, and out of htis layer
#        return getL(self.state)

#cimport GlobalState
#import GlobalState

cdef PyTorch.GlobalState globalState = PyTorch.getGlobalState()

#require(PyTorch.globalState.L, 'cltorch')

cdef class ClGlobalState(object):
##    cdef lua_State *L
##    cdef THGenerator *generator
    cdef THClState *state

    def __cinit__(ClGlobalState self):
        print('ClGlobalState.__cinit__')

    def __dealloc__(self):
        print('ClGlobalState.__dealloc__')

cdef ClGlobalState clGlobalState

def init():
    global clGlobalState
    cdef THClState *state2
    print('initializing PyClTorch...')
    require(globalState.L, 'cltorch')
    clGlobalState = ClGlobalState()
    clGlobalState.state = getState(globalState.L)
##    globalState.L = 
##    globalState.generator = <THGenerator *>(getGlobal(globalState.L, 'torch', '_gen'))
    print('state null:', clGlobalState.state == NULL)
    print('state: ', hex(<long>(clGlobalState.state)))
#    state2 = getState(globalState.L)
    print(' ... PyClTorch initialized')

init()

