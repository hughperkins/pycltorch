from floattensor import FloatTensor

import PyClTorch

def cl(self):
    print('cl')
    res = PyClTorch.FloatTensorToClTensor(self)
    print('res', res)
    return res

FloatTensor.cl = cl

