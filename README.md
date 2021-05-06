# Ultra

The main goal of Ultra is speed up prediction for PyTorch based models.

![alt text](https://github.com/edvgha/Ultra/blob/main/docs/flow.png?raw=true)

## Build

From project root do : 
1. For the first time fetch and extract libtorch in 'ext' directory by running **get_libtorch.sh**
2. **mkdir build ; cd build**
3. **cmake ..**
4. **make -j**
5. **make test (optional)**

## Demo 
Currently there are two demos **DeepAndWide** and **Resnet18**
1. To run **DeepAndWide** demo from build directory run DeepAndWide/synthetic_run_daw executable.
   The demo will check functional correctness, run both **Generated** and **PyTorch** forwards 5000 time measure runtime and display.
2. To run **Resnet18** demo from build directory run Resnet18/synthetic_run_res18 executable.
   The demo will check functional correctness, run both **Generated** and **PyTorch** forwards 5 time measure runtime and display.

## Install 
TODO

## Getting Started 
TODO

## Limitations

Currently build only tested on macOS Big Sur Version 11.3 and supported libtorch library version is 1.7.1
