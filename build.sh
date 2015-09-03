#!/bin/bash

# - must have installed torch, and activated
# - must have installed cltorch and clnn:
#    luarocks install cltorch
#    luarocks install clnn
# - must have prerequisites for pytorch installed, ie Cython, numpy, etc
# - pytorch must be cloned to a sibling folder of this folder, ie the parent of pycltorch should also contain pytorch

if [[ x${NOPYTORCHBUILD} == x ]]; then {
    cd ../pytorch
    git checkout master
    git pull
    pip uninstall -y PyTorch
    ./build.sh || exit 1
    python setup.py install || exit 1
} fi
rm -Rf build cbuild PyClBuild.so dist *.so
#mkdir cbuild
#(cd cbuild; cmake .. && make -j 4 ) || exit 1
python setup.py build_ext -i || exit 1

