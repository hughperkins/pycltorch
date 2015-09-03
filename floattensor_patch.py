from floattensor import FloatTensor

import PyClTorch

def cl(self):
    print('cl')
    print('self', self)
    res = PyClTorch.FloatTensorToClTensor(self)
    print('res', res)
    return res

#def newfunction(self):
#    print('newfunction')

#PyTorch.newfunction = newfunction

FloatTensor.cl = cl

