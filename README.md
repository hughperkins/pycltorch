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
 ... PyClTorch initialized
a 0.785071 0.192063
0.335184 0.825957
0.26625 0.501567
[torch.FloatTensor of size 3x2]

cl
Using NVIDIA Corporation , OpenCL platform: NVIDIA CUDA
Using OpenCL device: GeForce 940M
('res', <PyClTorch.ClTensor object at 0x7f064f64e150>)
<class 'floattensor.FloatTensor'>
<type 'PyClTorch.ClTensor'>
sum: 2.90609288216
```

