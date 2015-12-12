from __future__ import print_function

from floattensor import FloatTensor, LongStorage

import PyClTorch
import PyTorchAug
#from PyTorchAug import *

def cl(self):
    print('cl')
    res = PyClTorch.FloatTensorToClTensor(self)
    print('res', res)
    return res

FloatTensor.cl = cl

#PyTorchAug.

#def Linear_cl(self):
#    print('Linear_cl')
#    print('self', self)
#    self.cl()
#    return self

## import PyTorch
#Linear.cl = Linear_cl

print('dir(PyTorchAug)', dir(PyTorchAug))
print('dir(PyClTorch)', dir(PyClTorch))
PyTorchAug.cythonClasses['torch.ClTensor'] = {'popFunction': PyClTorch.cyPopClTensor}
PyTorchAug.populateLuaClassesReverse()

PyTorchAug.pushFunctionByPythonClass[PyClTorch.ClTensor] = PyClTorch.cyPushClTensor

