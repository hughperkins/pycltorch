# pycltorch
POC for Python wrappers for cltorch/clnn

# Example usage

## Pre-requisites

* Have installed torch, per https://github.com/torch/distro
* Have installed cltorch and clnn:
```
luarocks install cltorch
luarocks install clnn
```
* Have python 2.7
* Have setup a virtualenv, with cython, and numpy:
```
virtualenv env
source env/bin/activate
pip install cython
pip install numpy
pip install Jinja2
```
* *You need to have pytorch repo next to pycltorch repository*, ie the parent folder of pycltorch should contain [pytorch](https://github.com/hughperkins/pytorch)

To run:
```
./build.sh
./run.sh
```

Example script:
```
from __future__ import print_function
import PyClTorch

import PyTorch
from PyTorchAug import *

def myeval(expr):
    print(expr, ':', eval(expr))

a = PyTorch.FloatTensor(4, 3).uniform()
print('a', a)
a = a.cl()
print(type(a))

print('a.dims()', a.dims())
print('a.size()', a.size())
print('a', a)

print('sum:', a.sum())

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

output = linear.forward(a)

print('output.dims()', output.dims())
print('output.size()', output.size())

outputFloat = output.float()
print('outputFloat', outputFloat)

print('output', output)
```

Output:
```
initializing PyTorch...
generator null: False
 ... PyTorch initialized
dir(PyTorchAug) ['ClassNLLCriterion', 'Linear', 'LogSoftMax', 'LuaClass', 'PyTorch', 'Sequential', '__builtins__', '__doc__', '__file__', '__loader__', '__name__', '__package__', 'cythonClasses', 'getNextObjectId', 'lua', 'luaClasses', 'luaClassesReverse', 'nextObjectId', 'popString', 'populateLuaClassesReverse', 'print_function', 'pushFunctionByPythonClass', 'pushGlobal', 'pushGlobalFromList', 'pushObject', 'registerObject', 'torchType', 'unregisterObject']
dir(PyClTorch) ['ClGlobalState', 'ClTensor', 'FloatTensorToClTensor', 'PyTorch', '__builtins__', '__doc__', '__name__', '__package__', 'array', 'cyPopClTensor', 'cyPushClTensor']
initializing PyClTorch...
 ... PyClTorch initialized
a 0.427133 0.215103 0.986279
0.0733506 0.145054 0.572549
0.744484 0.953438 0.402136
0.543194 0.140102 0.868887
[torch.FloatTensor of size 4x3]

cl
Using NVIDIA Corporation , OpenCL platform: NVIDIA CUDA
Using OpenCL device: GeForce 940M
res 0.427133 0.215103 0.986279
0.0733506 0.145054 0.572549
0.744484 0.953438 0.402136
0.543194 0.140102 0.868887
[torch.ClTensor of size 4x3]

<type 'PyClTorch.ClTensor'>
a.dims() 2
a.size() 4 3
[torch.FloatTensor of size 2]

a 0.427133 0.215103 0.986279
0.0733506 0.145054 0.572549
0.744484 0.953438 0.402136
0.543194 0.140102 0.868887
[torch.ClTensor of size 4x3]

sum: 6.07170915604
b.dims() -1
b.size() None
b [torch.ClTensor with no dimension]

cl
res [torch.ClTensor with no dimension]

c.dims() -1
c.size() None
c [torch.ClTensor with no dimension]

creating Linear...
Linear.__init__
created linear
linear: nn.Linear(3 -> 5)
linear.output : [torch.FloatTensor with no dimension]

linear.output.dims() : 0
linear.output.size() : None
linear.output.nElement() : 0
Linear.__init__
type(linear) : <class 'PyTorchAug.Linear'>
type(linear.output) : <type 'PyClTorch.ClTensor'>
linear.output.dims() : 0
linear.output.size() : [torch.FloatTensor with no dimension]

linear.output : [torch.ClTensor with no dimension]

output.dims() 2
output.size() 4 5
[torch.FloatTensor of size 2]

outputFloat 0.387531 -0.367914 0.152106 -0.463973 0.0765648
0.289818 -0.140128 0.330504 -0.334515 -0.128896
-0.0304782 -0.868041 0.362495 -0.864463 0.59697
0.329106 -0.378436 0.173141 -0.489081 0.105022
[torch.FloatTensor of size 4x5]

output 0.387531 -0.367914 0.152106 -0.463973 0.0765648
0.289818 -0.140128 0.330504 -0.334515 -0.128896
-0.0304782 -0.868041 0.362495 -0.864463 0.59697
0.329106 -0.378436 0.173141 -0.489081 0.105022
[torch.ClTensor of size 4x5]
```

