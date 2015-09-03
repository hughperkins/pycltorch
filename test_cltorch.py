from __future__ import print_function
import PyClTorch

#PyClTorch.newfunction(123)

import PyTorch
from PyTorchAug import *

#a = PyTorch.foo(3,2)
#print('a', a)
#print(PyTorch.FloatTensor(3,2))

a = PyTorch.FloatTensor(3, 2).uniform()
print('a', a)
acl = a.cl()
print(type(a))
print(type(acl))

print('sum:', acl.sum())

linear = Linear(3,5)
print('linear:', linear)

linearCl = linear.cl()
print('linearCl', linearCl)
print('linearCl.output', linearCl.output)

