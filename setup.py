# Copyright Hugh Perkins 2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import print_function
import os
import os.path
from os.path import join
import platform
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

home_dir = os.getenv('HOME')
print('home_dir:', home_dir)

torch_install_dir = os.getenv('TORCH_INSTALL')
if torch_install_dir is None:
    raise Exception('Please ensure TORCH_INSTALL env var is defined')
osfamily = platform.uname()[0]
print('torch_install:', torch_install_dir)
print('os family', osfamily)

cython_present = True

compile_options = []
osfamily = platform.uname()[0]
if osfamily == 'Windows':
    compile_options.append('/EHsc')
elif osfamily == 'Linux':
    compile_options.append('-std=c++0x')
    compile_options.append('-g')
    if 'DEBUG' in os.environ:
        compile_options.append('-O0')
else:
    pass
    # put other options etc here if necessary

runtime_library_dirs = []
libraries = []
# libraries.append('mylib')
# libraries.append('clnnWrapper')
libraries.append('TH')
libraries.append('THCl')
libraries.append('PyTorchNative')
# libraries.append('PyTorch')
library_dirs = []
library_dirs.append('cbuild')
library_dirs.append(join(torch_install_dir, 'lib'))

if osfamily == 'Linux':
    runtime_library_dirs = ['.']

if osfamily == 'Windows':
    libraries.append('winmm')

sources = ["PyClTorch.cxx", "clnnWrapper.cpp"]
if cython_present:
    sources = ["PyClTorch.pyx", "clnnWrapper.cpp"]
ext_modules = [
    Extension("PyClTorch",
              sources=sources,
              include_dirs=[
                  join(torch_install_dir, 'include'),
                  join(torch_install_dir, 'include/TH'),
                  join(torch_install_dir, 'include/THCl'),
                  '/usr/include/lua5.1',
                  '../pytorch/src'],
              library_dirs=library_dirs,
              libraries=libraries,
              extra_compile_args=compile_options,
              runtime_library_dirs=runtime_library_dirs,
              language="c++"),
    #    Extension("GlobalState",
    #              sources=['GlobalState.pyx'],
    #              include_dirs=[
    #                  home_dir + '/torch/install/include/TH',
    #                  home_dir + '/torch/install/include/THCl',
    #                  '../pytorch'],
    #              library_dirs=library_dirs,
    #              libraries=libraries,
    #              extra_compile_args=compile_options,
    #              runtime_library_dirs=runtime_library_dirs,
    #              language="c++"),
]

ext_modules = cythonize(ext_modules)

setup(
    name='PyClTorch',
    version='SNAPSHOT',
    author="Hugh Perkins",
    author_email="hughperkins@gmail.com",
    description=(
        'Python wrappers for cltorch'),
    license='BSD2',
    url='https://github.com/hughperkins/pycltorch',
    long_description='',
    classifiers=[
    ],
    install_requires=[],
    scripts=[],
    ext_modules=ext_modules,
)
