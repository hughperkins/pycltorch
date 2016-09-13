from __future__ import print_function
import PyClTorch
from PyTorchAug import nn

# PyClTorch.newfunction(123)

import PyTorch
from PyTorchAug import *


def myeval(expr):
    print(expr, ':', eval(expr))


if __name__ == '__main__':
    # a = PyTorch.foo(3,2)
    # print('a', a)
    # print(PyTorch.FloatTensor(3,2))

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
    print('got b')
    myeval('b')
    b.resizeAs(a)
    myeval('b')
    print('run uniform')
    b.uniform()
    myeval('b')

    print('create new b')
    b = PyClTorch.ClTensor()
    print('b.dims()', b.dims())
    print('b.size()', b.size())
    print('b', b)

    c = PyTorch.FloatTensor().cl()
    print('c.dims()', c.dims())
    print('c.size()', c.size())
    print('c', c)

    print('creating Linear...')
    linear = nn.Linear(3, 5)
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
    # print('linearCl.output', linear.output)

    output = linear.forward(a)

    print('output.dims()', output.dims())
    print('output.size()', output.size())

    outputFloat = output.float()
    print('outputFloat', outputFloat)

    print('output', output)

    mlp = nn.Sequential()
    mlp.add(nn.SpatialConvolutionMM(1, 16, 5, 5, 1, 1, 2, 2))
    mlp.add(nn.ReLU())
    mlp.add(nn.SpatialMaxPooling(3, 3, 3, 3))
    mlp.add(nn.SpatialConvolutionMM(16, 32, 5, 5, 1, 1, 2, 2))
    mlp.add(nn.ReLU())
    mlp.add(nn.SpatialMaxPooling(2, 2, 2, 2))
    mlp.add(nn.Reshape(32 * 4 * 4))
    mlp.add(nn.Linear(32 * 4 * 4, 150))
    mlp.add(nn.Tanh())
    mlp.add(nn.Linear(150, 10))
    mlp.add(nn.LogSoftMax())

    mlp.cl()

    print('mlp', mlp)
    myeval('mlp.output')
    input = PyTorch.FloatTensor(128, 1, 28, 28).uniform().cl()
    myeval('input[0]')
    output = mlp.forward(input)
    myeval('output[0]')
