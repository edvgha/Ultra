#!/bin/bash
LIB_TORCH_PATH="../../../../ext/libtorch"
# Run to generate gout and generate 'forward' function 
./$1
# Max number of subdirectories
END=5
for ((i=1;i<=END;i++)); do
  if [ ! -d "$i" ] 
  then
    # Dir not exists
    break
  fi
  cd $i
  # Compile generated forward function and link with 'main'
  g++ test_main.cc forward.cpp -o out -I $LIB_TORCH_PATH/include -L $LIB_TORCH_PATH/lib -ltorch_cpu -lc10 -std=c++17
  # Load libtorch
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_TORCH_PATH/lib
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
    cd -
    rm -rf $i
    exit 1
  fi
  # Clean up
  cd -
  rm -rf $i
done
