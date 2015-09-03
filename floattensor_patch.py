from floattensor import FloatTensor, Linear

import PyClTorch

def cl(self):
    print('cl')
    res = PyClTorch.FloatTensorToClTensor(self)
    print('res', res)
    return res

FloatTensor.cl = cl

def Linear_cl(self):
    print('Linear_cl')
    print('self', self)
    return self

# import PyTorch
Linear.cl = Linear_cl

