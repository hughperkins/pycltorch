from __future__ import print_function
import PyClTorch

#PyClTorch.newfunction(123)

import PyTorch

#a = PyTorch.foo(3,2)
#print('a', a)
#print(PyTorch.FloatTensor(3,2))

a = PyTorch.FloatTensor(3, 2)
print('a', a)
a.cl()
