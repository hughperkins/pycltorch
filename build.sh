#!/bin/bash

# - must have installed torch, and activated
# - must have installed cltorch and clnn:
#    luarocks install cltorch
#    luarocks install clnn
# - must have installed PyTorch

mkdir cbuild
(cd cbuild; cmake .. && make -j 4 ) || exit 1
rm -Rf build PyClBuild.so
python setup.py build_ext -i || exit 1

