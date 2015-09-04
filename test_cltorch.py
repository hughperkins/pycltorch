from __future__ import print_function
import PyClTorch

#PyClTorch.newfunction(123)

import PyTorch
from PyTorchAug import *

def myeval(expr):
    print(expr, ':', eval(expr))

#a = PyTorch.foo(3,2)
#print('a', a)
#print(PyTorch.FloatTensor(3,2))

a = PyTorch.FloatTensor(4, 3).uniform()
print('a', a)
a = a.cl()
print(type(a))

print('a.dims()', a.dims())
print('a.size()', a.size())
print('a', a)

print('sum:', a.sum())
myeval('a + 1')

b = PyClTorch.ClTensor()
print('b.dims()', b.dims())
print('b.size()', b.size())
print('b', b)

c = PyTorch.FloatTensor().cl()
print('c.dims()', c.dims())
print('c.size()', c.size())
print('c', c)

print('creating Linear...')
linear = Linear(3,5)
print('created linear')
print('linear:', linear)
myeval('linear.output')
myeval('linear.output.dims()')
myeval('linear.output.size()')
myeval('linear.output.nElement()')

linear = linear.cl()
myeval('type(linear)')
myeval('type(linear.output)')
myeval('linear.output.dims()')
myeval('linear.output.size()')
myeval('linear.output')
#print('linearCl.output', linear.output)

output = linear.forward(a)

print('output.dims()', output.dims())
print('output.size()', output.size())

outputFloat = output.float()
print('outputFloat', outputFloat)

print('output', output)



