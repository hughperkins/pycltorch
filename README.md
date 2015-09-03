# pycltorch
POC for Python wrappers for cltorch/clnn

# Latest news

* can instantiate a ClTensor now :-)  And run `sumall()` :-)

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
* *You need to have pytorch repo next to pycltorch repository*, ie the parent folder of pycltorch should contain [pytorch](https://github.com/hughperkins/pytorch)  This requirement will probably change in the future, to requiring that pytorch is already installed, but it makes for fast development

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

a = PyTorch.FloatTensor(3, 2).uniform()
print('a', a)
acl = a.cl()
print(type(a))
print(type(acl))

print('sum:', acl.sum())
```

Output:
```
initializing PyTorch...
GlobalState.__cinit__
loaded lua library
generator null: False
 ... PyTorch initialized
initializing PyClTorch...
cltorch init.lua
cltorch._state	userdata: 0x2c59440
ClGlobalState.__cinit__
state: 46502976
state null: False
state:  0x2c59440
 ... PyClTorch initialized
a 0.0242427 0.866875
0.122811 0.855529
0.380983 0.891376
[torch.FloatTensor of size 3x2]

cl
('self', 0.0242427 0.866875
0.122811 0.855529
0.380983 0.891376
[torch.FloatTensor of size 3x2]
)
ClTensor.__cinit__
Using NVIDIA Corporation , OpenCL platform: NVIDIA CUDA
Using OpenCL device: GeForce 940M
('res', <PyClTorch.ClTensor object at 0x7ff37405b150>)
<class 'floattensor.FloatTensor'>
<type 'PyClTorch.ClTensor'>
sum: 3.14181804657
ClTensor.__dealloc__
```

