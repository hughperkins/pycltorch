from __future__ import print_function
import cython
cimport cython
cimport cpython.array
import array
import PyTorch
from PyTorch import _LongStorage, _FloatTensor
cimport PyTorch
cimport Storage

# {% set Real = 'Cl' %}
# {% set real = 'cl' %}


cdef extern from "LuaHelper.h":
    cdef struct lua_State
    void *getGlobal(lua_State *L, const char *name1, const char *name2);
    void luaRequire(lua_State *L, const char *name)


cdef extern from "THClGeneral.h":
    cdef struct THClState
    int THClState_getNumDevices(THClState* state);
    void THClState_setDevice(THClState* state, int device);
    int THClState_getDevice(THClState* state)


cdef extern from "THStorage.h":
    cdef struct THLongStorage
    void THLongStorage_free(THLongStorage *self)


cdef extern from "THTensor.h":
    cdef struct THFloatTensor
    THFloatTensor *THFloatTensor_new()
    void THFloatTensor_resize1d(THFloatTensor *self, long size0)
    void THFloatTensor_resize2d(THFloatTensor *self, long size0, long size1)
    void THFloatTensor_set1d(const THFloatTensor *tensor, long x0, float value)
    void THFloatTensor_set2d(const THFloatTensor *tensor, long x0, long x1, float value)
    void THFloatTensor_free(THFloatTensor *self)
    THLongStorage *THFloatTensor_newSizeOf(THFloatTensor *self)
    THLongStorage *THFloatTensor_newStrideOf(THFloatTensor *self)


cdef extern from "THClTensor.h":
    cdef struct THClTensor
    THClTensor *THClTensor_newv2(THClState *state, int device)
    THClTensor *THClTensor_newWithSize1d(THClState *state, int device, long size0)
    THClTensor *THClTensor_newWithSize2d(THClState *state, int device, long size0, long size1)
    THClTensor *THClTensor_newWithSize3d(THClState *state, int device, long size0, long size1, long size2)
    THClTensor *THClTensor_newWithSize4d(THClState *state, int device, long size0, long size1, long size2, long size3)
    float THClTensor_get1d(THClState *state, const THClTensor *tensor, long x0)
    float THClTensor_get2d(THClState *state, const THClTensor *tensor, long x0, long x1)
    void THClTensor_retain(THClState *state, THClTensor*self)
    void THClTensor_free(THClState *state, THClTensor *tensor)
    int THClTensor_nDimension(THClState *state, THClTensor *tensor)
    long THClTensor_size(THClState *state, const THClTensor *self, int dim)
    long THClTensor_nElement(THClState *state, const THClTensor *self)
    void THClTensor_resizeAs(THClState *state, THClTensor *self, THClTensor *model)
    THClTensor *THClTensor_newSelect(THClState *state, THClTensor *self, int dimension, int sliceIndex)
    void THClTensor_resize1d(THClState *state, THClTensor *self, long size0)
    void THClTensor_resize2d(THClState *state, THClTensor *self, long size0, long size1)
    void THClTensor_resize(THClState *state, THClTensor *tensor, THLongStorage *size, THLongStorage *stride)


cdef extern from "THClTensorCopy.h":
    void THClTensor_copyFloat(THClState *state, THClTensor *self, THFloatTensor *src)
    void THFloatTensor_copyCl(THClState *state, THFloatTensor *self, THClTensor *src)


cdef extern from "THClTensorMath.h":
    float THClTensor_sumall(THClState *state, THClTensor *self)
    void THClTensor_add(THClState *state, THClTensor *res, THClTensor *self, float scalar)


cdef extern from "clnnWrapper.h":
    THClState *getState(lua_State *L)
    THClTensor *popClTensor(lua_State *L)
    void pushClTensor(THClState *state, lua_State *L, THClTensor *tensor)


def cyPopClTensor():
    cdef THClTensor *tensorC = popClTensor(globalState.L)
    cdef ClTensor tensor = ClTensor_fromNative(tensorC)
    return tensor


def cyPushClTensor(ClTensor tensor):
    pushClTensor(clGlobalState.state, globalState.L, tensor.native)


cdef class ClTensor(object):
    cdef THClTensor *native

    def __cinit__(ClTensor self, *args, _allocate=True):
        if _allocate:
            if len(args) == 1 and isinstance(args[0], _LongStorage):  # it's a size tensor
               print('longstorage sie tensor')
               self.native = THClTensor_newv2(clGlobalState.state, 0) #FIXME get device from state
               self.resize(args[0])
               return
            if len(args) == 1 and isinstance(args[0], list):  # it's some data
                print('data list')
                node = args[0]
                nDims = 0
                dims = []
                nElements = 1 # len(node)
                # dims.append(len(node))
                while isinstance(node, list):
                    print('node', node)
                    nElements *= len(node)
                    dims.append(len(node))
                    nDims += 1
                    node = node[0]
                print('nDims', nDims, 'nElements', nElements, 'dims', dims)
                if nDims > 2:
                    raise Exception('more than 2 dimensions in list initializer not currently handled')
                size = PyTorch.LongStorage(nDims)
                for i, dim in enumerate(dims):
                    size[i] = dim
                print('size', size)
                floatnative = THFloatTensor_new()
                self.native = THClTensor_newv2(clGlobalState.state, 0) #FIXME get device from state
                if nDims == 2:
                    THFloatTensor_resize2d(floatnative, dims[0], dims[1])
                    THClTensor_resize2d(clGlobalState.state, self.native, dims[0], dims[1])
                    for i1 in range(dims[0]):
                        row = args[0][i1]
                        for i2 in range(dims[1]):
                            value = row[i2]
                            THFloatTensor_set2d(floatnative, i1, i2, value)
                elif nDims == 1:
                    THFloatTensor_resize1d(floatnative, dims[0])
                    THClTensor_resize1d(clGlobalState.state, self.native, dims[0])
                    for i1 in range(dims[0]):
                        value = args[0][i1]
                        THFloatTensor_set1d(floatnative, i1, value)
                # self.resize(args[0])
                THClTensor_copyFloat(clGlobalState.state, self.native, floatnative)
                # self.native.copy(floatnative)
                THFloatTensor_free(floatnative)
                return
            for arg in args:
                if not isinstance(arg, int):
                    raise Exception('cannot provide arguments to initializer')
            device = THClState_getDevice(clGlobalState.state)
            if len(args) == 0:
                self.native = THClTensor_newv2(clGlobalState.state, device)
            elif len(args) == 1:
                self.native = THClTensor_newWithSize1d(clGlobalState.state, device, args[0])
            elif len(args) == 2:
                self.native = THClTensor_newWithSize2d(clGlobalState.state, device, args[0], args[1])
            elif len(args) == 3:
                self.native = THClTensor_newWithSize3d(clGlobalState.state, device, args[0], args[1], args[2])
            elif len(args) == 4:
                self.native = THClTensor_newWithSize4d(clGlobalState.state, device, args[0], args[1], args[2], args[3])
            else:
                raise Exception('Not implemented, len(args)=' + str(len(args)))

    def __dealloc__(ClTensor self):
        THClTensor_free(clGlobalState.state, self.native)

    @staticmethod
    def new():
        return ClTensor()

    cpdef float get1d(self, int x0):
        return THClTensor_get1d(clGlobalState.state, self.native, x0)

    cpdef float get2d(self, int x0, int x1):
        return THClTensor_get2d(clGlobalState.state, self.native, x0, x1)

    def __repr__(ClTensor self):
        cdef PyTorch._FloatTensor floatTensor = self.float()
        floatRepr = floatTensor.__repr__()
        clRepr = floatRepr.replace('FloatTensor', 'ClTensor')
        return clRepr

    def float(ClTensor self):
        cdef PyTorch._FloatTensor floatTensor = PyTorch._FloatTensor.new()
        cdef Storage._LongStorage size = self.size()
        if size is None:
            return PyTorch._FloatTensor()
        if len(size) == 0:
            return PyTorch._FloatTensor()
        floatTensor.resize(size)
        THFloatTensor_copyCl(clGlobalState.state, floatTensor.native, self.native)
        return floatTensor

    def copy(ClTensor self, PyTorch._FloatTensor src):
        THClTensor_copyFloat(clGlobalState.state, self.native, src.native)
        return self

    cpdef int dims(ClTensor self):
        return THClTensor_nDimension(clGlobalState.state, self.native)

    def size(ClTensor self):
        cdef int dims = self.dims()
        cdef Storage._LongStorage size
        if dims >= 0:
            size = Storage._LongStorage(dims)
            for d in range(dims):
                size[d] = THClTensor_size(clGlobalState.state, self.native, d)
            return size
        else:
            return None  # not sure how to handle this yet

    def nElement(ClTensor self):
        return THClTensor_nElement(clGlobalState.state, self.native)

    def sum(ClTensor self):
        return THClTensor_sumall(clGlobalState.state, self.native)

    def __add__(ClTensor self, float scalar):
        cdef ClTensor res = ClTensor()
        THClTensor_add(clGlobalState.state, res.native, self.native, scalar)
        return res

    def __getitem__(ClTensor self, int index):
        if self.dims() == 1:
            return self.get1d(index)
        cdef THClTensor *res = THClTensor_newSelect(clGlobalState.state, self.native, 0, index)
        return ClTensor_fromNative(res, False)

    def __setitem__(ClTensor self, int index, float value):
        if self.dims() == 1:
            self.set1d(index, value)
        else:
            raise Exception("not implemented")

#    def __getitem__(ClTensor self, int index):
#        if self.dims() == 1:
#            return self.get1d(index)
#        cdef THClTensor *res = THClTensor_newSelect(clGlobalState.state, self.native, 0, index)
#        return ClTensor_fromNative(res, False)

    def resizeAs(ClTensor self, model):
        cdef ClTensor model_cl
        cdef PyTorch._FloatTensor model_float
        if isinstance(model, ClTensor):
            model_cl = model
            THClTensor_resizeAs(clGlobalState.state, self.native, model_cl.native)
            return self
        elif isinstance(model, PyTorch._FloatTensor):
            model_float = model
            sizenative = THFloatTensor_newSizeOf(model_float.native)
            stridenative = THFloatTensor_newStrideOf(model_float.native)
            THClTensor_resize(clGlobalState.state, self.native, sizenative, stridenative)
            THLongStorage_free(sizenative)
            THLongStorage_free(stridenative)
            return self
        else:
            raise Exception('resizeAs not implemetned for %s' % type(model))

    def uniform(ClTensor self, float a=0, float b=1):
        cdef Storage._LongStorage size = self.size()
        cdef PyTorch._FloatTensor floatTensor
        floatTensor = PyTorch._FloatTensor(size)
        floatTensor.uniform(a, b)
        self.copy(floatTensor)
        return self


cdef ClTensor_fromNative(THClTensor *tensorC, retain=True):
    cdef ClTensor tensor = ClTensor(_allocate=False )
    tensor.native = tensorC
    if retain:
        THClTensor_retain(clGlobalState.state, tensorC)
    return tensor


def FloatTensorToClTensor(PyTorch._FloatTensor floatTensor):
    cdef Storage._LongStorage size = floatTensor.size()
    cdef ClTensor clTensor
    cdef int nElement = floatTensor.nElement()
    if nElement > 0:
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
    else:
        return ClTensor()


import floattensor_patch


cdef PyTorch.GlobalState globalState = PyTorch.getGlobalState()


cdef class ClGlobalState(object):
    cdef THClState *state


cdef ClGlobalState clGlobalState


def getDeviceCount():
    global clGlobalState
    return THClState_getNumDevices(clGlobalState.state)


def setDevice(device):
    global clGlobalState
    THClState_setDevice(clGlobalState.state, device)


def getDevice():
    global clGlobalState
    return THClState_getDevice(clGlobalState.state)


def init():
    global clGlobalState
    cdef THClState *state2
    print('initializing PyClTorch...')
    luaRequire(globalState.L, 'cltorch')
    luaRequire(globalState.L, 'clnn')
    clGlobalState = ClGlobalState()
    clGlobalState.state = getState(globalState.L)
    print(' ... PyClTorch initialized')

init()
