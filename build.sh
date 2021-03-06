#!/bin/bash

# - must have installed torch, and activated
# - must have installed cltorch and clnn:
#    luarocks install cltorch
#    luarocks install clnn
# - must have prerequisites for pytorch installed, ie Cython, numpy, etc
# - pytorch must be cloned to a sibling folder of this folder, ie the parent of pycltorch should also contain pytorch

# Availalbe env vars:
# NOPYTORCHBUILD=1  => doesnt rebuild pytorch
# NOGIT=1  => doesnt pull from git into pytorch

export TORCH_INSTALL=$(dirname $(dirname $(which luajit) 2>/dev/null) 2>/dev/null)

if [[ x${NOPYTORCHBUILD} == x ]]; then {
    (
        cd ../pytorch
        if [[ x${NOGIT}} == x ]]; then {
            git checkout master
            git pull
        } fi
        pip uninstall -y PyTorch
        ./build.sh || exit 1
        python setup.py install || exit 1
    )
} fi
rm -Rf build cbuild dist *.so *.pyc PyCltorch.cpp __pycache__
#mkdir cbuild
#(cd cbuild; cmake .. && make -j 4 ) || exit 1
# python setup.py build_ext -i || exit 1
pip install --verbose -e ./
