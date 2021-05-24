#!/bin/bash
# Load LibTorch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../../ext/libtorch/lib
# Run to generate gout and generate 'forward' function 
./unit_test
# Compile generated forward function and link with 'main'
g++ test_main.cc forward.cpp -o out -I ../../../ext/libtorch/include -L ../../../ext/libtorch/lib -ltorch_cpu -lc10 -std=c++17
# Get output
./out > test_out
# Output from generated function
FILE1=test_out
# Golden output
FILE2=test_gout
if cmp --silent -- "$FILE1" "$FILE2"; then
  echo "Files contents are identical"
else
  echo "Files differ"
  # Clean up
  rm forward.cpp forward.h test_gout test_out out test_main.cc
  exit 1
fi
# Clean up
rm forward.cpp forward.h test_gout test_out out test_main.cc
