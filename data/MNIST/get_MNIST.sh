#!/bin/bash
files=(
    train-images-idx3-ubyte.gz 
    train-labels-idx1-ubyte.gz 
    t10k-images-idx3-ubyte.gz 
    t10k-labels-idx1-ubyte.gz
)

for i in files
    `wget http://yann.lecun.com/exdb/mnist/${i}`
done 