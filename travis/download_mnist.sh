#!/bin/bash

echo downloading mnist data
mkdir -p data/mnist
(cd data/mnist
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O train-images-idx3-ubyte.gz
    gunzip train-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O train-labels-idx1-ubyte.gz
    gunzip train-labels-idx1-ubyte.gz
)
