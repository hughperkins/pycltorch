from floattensor import FloatTensor

def cl(self):
    print('cl')
    print('self', self)

#def newfunction(self):
#    print('newfunction')

#PyTorch.newfunction = newfunction

FloatTensor.cl = cl

