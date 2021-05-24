#!/bin/bash

## Check existence of libtorch
if [ ! -d "ext/libtorch" ] 
then
    echo "Downloading libtorch ..."
    rm -rf ext
    mkdir ext
    cd ext
    wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.7.1.zip
    echo "Extracting libtorch ..."
    tar xvzf libtorch-macos-1.7.1.zip
    cd -
fi
