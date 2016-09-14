from __future__ import print_function
import PyTorch
import PyClTorch
import os
from PyTorchAug import nn
from PyTorchAug import *
from test.test_helpers import myeval


def test_cltorch():
    if 'ALLOW_NON_GPUS' in os.environ:
        PyClTorch.setAllowNonGpus(True)

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
    linear = nn.Linear(3, 5).float()
    print('created linear')
    print('linear:', linear)
    myeval('linear.output')
    myeval('linear.output.dims()')
    myeval('linear.output.size()')
    myeval('linear.output.nElement()')

    linear_cl = linear.clone().cl()
    print('type(linear.output)', type(linear.output))
    print('type(linear_cl.output)', type(linear_cl.output))
    assert str(type(linear.output)) == '<class \'PyTorch._FloatTensor\'>'
    assert str(type(linear_cl.output)) == '<class \'PyClTorch.ClTensor\'>'
    # myeval('type(linear)')
    # myeval('type(linear.output)')
    myeval('linear_cl.output.dims()')
    myeval('linear_cl.output.size()')
    # myeval('linear.output')
    assert str(type(linear)) == '<class \'PyTorchAug.Linear\'>'
    assert str(type(linear_cl)) == '<class \'PyTorchAug.Linear\'>'
    # assert str(type(linear.output)) == '<class \'PyClTorch.ClTensor\'>'
    # assert linear.output.dims() == -1  # why is this 0? should be -1???
    # assert linear.output.size() is None  # again, should be None?

    a_cl = PyClTorch.ClTensor(4, 3).uniform()
    # print('a_cl', a_cl)
    output_cl = linear_cl.forward(a_cl)
    # print('output', output)
    assert str(type(output_cl)) == '<class \'PyClTorch.ClTensor\'>'
    assert output_cl.dims() == 2
    assert output_cl.size()[0] == 4
    assert output_cl.size()[1] == 5

    a = a_cl.float()
    output = linear.forward(a)
    assert str(type(output)) == '<class \'PyTorch._FloatTensor\'>'
    assert output.dims() == 2
    assert output.size()[0] == 4
    assert output.size()[1] == 5
    print('a.size()', a.size())
    print('a_cl.size()', a_cl.size())
    assert a[1][0] == a_cl[1][0]
    assert a[2][1] == a_cl[2][1]

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
    mlp.float()

    mlp_cl = mlp.clone().cl()

    print('mlp_cl', mlp_cl)
    # myeval('mlp.output')
    input = PyTorch.FloatTensor(128, 1, 28, 28).uniform()
    input_cl = PyClTorch.FloatTensorToClTensor(input.clone())  # This is a bit hacky...

    output = mlp.forward(input)
    # myeval('input[0]')
    output_cl = mlp_cl.forward(input_cl)
    # myeval('output[0]')

    assert (output_cl.float() - output).abs().max() < 1e-4
