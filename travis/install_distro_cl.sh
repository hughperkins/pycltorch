#!/bin/bash

# assume current directory is root of cloned distro-cl tree (and without cloning recursive)

BASE=$(PWD)
for pkg in cudnn cunn cunnx cutorch qttorch graph sdl2 threads submodule graphicsmagick audio dok argcheck fftw3 signal nnx qtlua gnuplot iTorch; do { sed -i -e "s/\(.*$pkg.*\)/echo skipping $pkg # \1/" install.sh; } done
awk ''NR==2{print "set -x"}1'' install.sh > ~install.sh
mv ~install.sh install.sh
chmod +x install.sh
cat install.sh
for pkg in opencl/cltorch opencl/clnn exe/luajit-rocks extra/nn pkg/cwrap pkg/paths pkg/sundown pkg/optim pkg/sys pkg/xlua pkg/image pkg/sys pkg/torch exe/trepl pkg/paths extra/penlight extra/lua-cjson extra/luaffifb extra/luafilesystem; do { git submodule update --init $pkg; } done
sed -i -e 's/$(MAKE)/$(MAKE) -j 4/' pkg/torch/rocks/torch-scm-1.rockspec
./install.sh -b
source ${BASE}/install/bin/torch-activate
luajit -l torch -e 'print("ok")'
luajit -l nn -e 'print("ok")'
luajit -l cltorch -e 'print("ok")'
luajit -l clnn -e 'print("ok")'
luajit -l optim -e 'print("ok")'
luajit -l torch -e 'torch.test()'
luajit -l nn -e 'nn.test()'
export TEST_EXCLUDES=test_blas,test_cumprod,test_cumsum,test_equals,test_indexcopy,test_indexfill,test_matrixwide,test_max2,test_mean,test_meanall,test_min2,test_norm,test_prod,test_prodall,test_sum_t,test_sumallt,test_reduceAll,test_sum,test_sum_t_offset,test_sumall
luajit -l cltorch -e "cltorch.setAllowNonGpus(1); cltorch.test()"
