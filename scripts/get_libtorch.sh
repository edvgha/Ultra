#!/bin/bash

## Check existence of libtorch
if [ ! -d "ext/libtorch" ] 
then
    echo "Downloading libtorch ..."
    rm -rf ext
    mkdir ext
    cd ext
    if [[ "$OSTYPE" == "linux-gnu"* ]]
    then
        # LINUX
        wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.7.1%2Bcpu.zip
        echo "Extracting libtorch ..."
        unzip libtorch-shared-with-deps-1.7.1+cpu.zip
    elif [[ "$OSTYPE" == "darwin"* ]]
    then
        # Mac OSX
        wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.7.1.zip
        echo "Extracting libtorch ..."
        tar xvzf libtorch-macos-1.7.1.zip
    else
 	echo "unknown OS type"
	exit
    fi
    cd -
fi
