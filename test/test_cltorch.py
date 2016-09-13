from __future__ import print_function
import PyTorch
import PyClTorch
from PyTorchAug import nn
from PyTorchAug import *
from test.test_helpers import myeval


def test_cltorch():
    # a = PyTorch.foo(3,2)
    # print('a', a)
    # print(PyTorch.FloatTensor(3,2))

    a = PyClTorch.ClTensor([3, 4, 9])
    assert a[0] == 3
    assert a[1] == 4
    assert a[2] == 9
    print('a', a)

    a = PyClTorch.ClTensor([[3, 5, 7], [9, 2, 4]])
    print('a', a)
    print('a[0]', a[0])
    print('a[0][0]', a[0][0])
    assert a[0][0] == 3
    assert a[1][0] == 9
    assert a[1][2] == 4

    PyTorch.manualSeed(123)
    a = PyTorch.FloatTensor(4, 3).uniform()
    print('a', a)
    a_cl = a.cl()
    print(type(a_cl))
    assert str(type(a_cl)) == '<class \'PyClTorch.ClTensor\'>'
    print('a_cl[0]', a_cl[0])
    print('a_cl[0][0]', a_cl[0][0])
    assert a[0][0] == a_cl[0][0]
    assert a[0][1] == a_cl[0][1]
    assert a[1][1] == a_cl[1][1]

    print('a.dims()', a.dims())
    print('a.size()', a.size())
    print('a', a)
    assert a.dims() == 2
    assert a.size()[0] == 4
    assert a.size()[1] == 3

    a_sum = a.sum()
    a_cl_sum = a_cl.sum()
    assert abs(a_sum - a_cl_sum) < 1e-4
    a_cl2 = a_cl + 3.2
    assert abs(a_cl2[1][0] - a[1][0] - 3.2) < 1e-4

    b = PyClTorch.ClTensor()
    print('got b')
    myeval('b')
    assert b.dims() == -1
    b.resizeAs(a)
    myeval('b')
    assert b.dims() == 2
    assert b.size()[0] == 4
    assert b.size()[1] == 3
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
    assert b.dims() == -1
    assert b.size() is None

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
