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
```

Output:
```
initializing PyTorch...
generator null: False
 ... PyTorch initialized
dir(PyTorchAug) ['ClassNLLCriterion', 'Linear', 'LogSoftMax', 'LuaClass', 'PyTorch', 'Sequential', '__builtins__', '__doc__', '__file__', '__loader__', '__name__', '__package__', 'cythonClasses', 'getNextObjectId', 'lua', 'luaClasses', 'luaClassesReverse', 'nextObjectId', 'popString', 'populateLuaClassesReverse', 'print_function', 'pushGlobal', 'pushGlobalFromList', 'pushObject', 'registerObject', 'torchType', 'unregisterObject']
dir(PyClTorch) ['ClGlobalState', 'ClTensor', 'FloatTensorToClTensor', 'PyTorch', '__builtins__', '__doc__', '__name__', '__package__', 'array', 'cyPopClTensor']
initializing PyClTorch...
 ... PyClTorch initialized
a 0.315532 0.875113
0.738289 0.577025
0.291994 0.509012
[torch.FloatTensor of size 3x2]

cl
Using NVIDIA Corporation , OpenCL platform: NVIDIA CUDA
Using OpenCL device: GeForce 940M
res <PyClTorch.ClTensor object at 0x7f42161d5168>
<class 'floattensor.FloatTensor'>
<type 'PyClTorch.ClTensor'>
sum: 3.30696415901
linear: nn.Linear(3 -> 5)
cythonClasses {'torch.ClTensor': {'popFunction': <built-in function cyPopClTensor>}, 'torch.FloatTensor': {'popFunction': <built-in function _popFloatTensor>}}
linearCl nn.Linear(3 -> 5)
linearCl.output <PyClTorch.ClTensor object at 0x7f42161d5180>
```

