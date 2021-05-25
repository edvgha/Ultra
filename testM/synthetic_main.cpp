#include "forward.h"
#include <torch/csrc/jit/serialization/import.h>
#include <chrono>

int main()
{
   Tensor x = ones({2, 2});
   {
      Tensor synthetic_out;
      std::cout << synthetic_forward(x) << std::endl;
   }
}
