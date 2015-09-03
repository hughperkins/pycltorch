#!/bin/bash

# - must have installed torch, and activated
# - must have installed cltorch and clnn:
#    luarocks install cltorch
#    luarocks install clnn
# - must have prerequisites for pytorch installed, ie Cython, numpy, etc
# - pytorch must be cloned to a sibling folder of this folder, ie the parent of pycltorch should also contain pytorch

if [[ x${NOPYTORCHBUILD} == x ]]; then { (cd ../pytorch; pip uninstall -y PyTorch; ./build.sh && python setup.py install) || exit 1; } fi
mkdir cbuild
(cd cbuild; cmake .. && make -j 4 ) || exit 1
rm -Rf build PyClBuild.so dist *.so
python setup.py build_ext -i || exit 1

