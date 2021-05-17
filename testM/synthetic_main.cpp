#include "forward.h"
#include <torch/csrc/jit/serialization/import.h>
#include <chrono>

int main()
{
   Tensor x = ones({2, 2});
   int a = 0;
   int b = 1;
   int c = -1;
   bool bl = true;
   bool oo = false;

   {
      Tensor synthetic_out;
      std::cout << synthetic_forward(x, bl, a) << std::endl;
      std::cout << synthetic_forward(x, oo, a) << std::endl;
      std::cout << synthetic_forward(x, bl, b) << std::endl;
      std::cout << synthetic_forward(x, oo, b) << std::endl;
      std::cout << synthetic_forward(x, bl, c) << std::endl;
      std::cout << synthetic_forward(x, oo, c) << std::endl;
   }
}
