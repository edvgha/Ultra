# Ultra

The main goal of Ultra is to implement the zero-overhead principle
by compiling dynamically specified data flow PyTorch JIT IR graph 
into C++ which allow to reduce DL framework overhead. 

       In general, C++ implementations obey the zero-overhead
       principle: What you don’t use, you don’t pay for. And
       further: What you do use, you couldn’t hand code any
       better.

![alt text](https://github.com/edvgha/Ultra/blob/main/docs/flow.png?raw=true)

## Build

From project root do : 
1. For the first time fetch and extract libtorch in 'ext' directory by running **get_libtorch.sh**
2. **mkdir build ; cd build**
3. **cmake ..**
4. **make -j**
5. **make test (optional)**

## Demo 
Currently there are two demos **DeepAndWide** and **LLD6**
1. To run **DeepAndWide** demo from build directory run DeepAndWide/synthetic_run_daw executable.
   The demo will check functional correctness, run both **Generated** and **PyTorch** forwards 5000 time measure runtime and display.
2. To run **LLD6** demo from build directory run LLD6/synthetic_run_lld6 executable.
   The demo will check functional correctness, run both **Generated** and **PyTorch** forwards 5000 time measure runtime and display.
3. To run **ResNet18** demo from build directory run Resnet18/synthetic_run_res18 executable.
   The demo will check functional correctness, run both **Generated** and **PyTorch** forwards 10 time measure runtime and display.
4. To run **ResNet50** demo from build directory run Resnet50/synthetic_run_res50 executable.
   The demo will check functional correctness, run both **Generated** and **PyTorch** forwards 10 time measure runtime and display.
   
<!-- tocstop -->

| Model | SpeedUp compared to PyTorch | Data |
| :---: | :---: | :---: |
| DeepAndWide | 40x | 64x1x32x50 |
| LLD6 | 20x | 32x128 |
| ResNet18 | 6x | 1x3x64x64 |
| ResNet50 | 6x | 1x3x64x64 |

Machine : 
 - Processor 3.2 GHz 6-Core Intel Core i7
 - Memory 32 GB 2667 MHz DDR4

## Install 
TODO

## Getting Started 
TODO

## Limitations

Currently build only tested on macOS Big Sur Version 11.3 and supported libtorch library version is 1.7.1
