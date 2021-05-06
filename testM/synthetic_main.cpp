#include "forward.h"
#include <torch/csrc/jit/serialization/import.h>
#include <chrono>

int main()
{
   Tensor x = ones({2, 2});
   int b = 0;

   {
      Tensor synthetic_out;
      std::cout << synthetic_forward(x, b) << std::endl;
      // std::cout << synthetic_forward(x, c) << std::endl;
   }
}
