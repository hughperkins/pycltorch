from __future__ import print_function
import PyClTorch

#PyClTorch.newfunction(123)

import PyTorch
from PyTorchAug import *

#a = PyTorch.foo(3,2)
#print('a', a)
#print(PyTorch.FloatTensor(3,2))

a = PyTorch.FloatTensor(4, 3).uniform()
print('a', a)
a = a.cl()
print(type(a))

print('sum:', a.sum())

linear = Linear(3,5)
print('linear:', linear)

linear = linear.cl()
print('linearCl.output', linear.output)

output = linear.forward(a)

