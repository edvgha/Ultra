#include "forward.h"
#include <torch/csrc/jit/serialization/import.h>
#include <chrono>

#define ITERS 5

c10::IValue pytorch_jit_forward(const std::vector<c10::IValue>& inputs)
{
   auto module = torch::jit::load("../Resnet18/resnet18.pt");
   module.eval();
   return module.forward(inputs);
}

int main()
{ 

   long long batch_size = 64;
   Tensor input = rand({batch_size, 3, 244, 244});

   {
      Tensor synthetic_out = synthetic_forward(input);
   
      IValue pytorch_jit_out = pytorch_jit_forward({input});

      auto expected = pytorch_jit_out.toTensor();
      auto expected_ptr = expected.data_ptr<float>();
      
      auto actual_ptr = synthetic_out.data_ptr<float>();

      TORCH_CHECK(expected.numel() == synthetic_out.numel(),  "BIZIMDIR");

      for (size_t i = 0; i < expected.numel(); ++i) 
      {
         if (std::abs(expected_ptr[i] - actual_ptr[i]) >= 0.0001) 
         {
            std::cout << "BIZIMDIR " << i << " expected: " << expected_ptr[i] << " actual: " << actual_ptr[i] << std::endl;
         }
      }
   }
   {
      Tensor synthetic_out = synthetic_forward(input);
   
      IValue pytorch_jit_out = pytorch_jit_forward({input});

      auto expected = pytorch_jit_out.toTensor();
      auto expected_ptr = expected.data_ptr<float>();
      
      auto actual_ptr = synthetic_out.data_ptr<float>();

      TORCH_CHECK(expected.numel() == synthetic_out.numel(),  "BIZIMDIR");

      for (size_t i = 0; i < expected.numel(); ++i) 
      {
         if (std::abs(expected_ptr[i] - actual_ptr[i]) >= 0.0001) 
         {
            std::cout << "BIZIMDIR " << i << " expected: " << expected_ptr[i] << " actual: " << actual_ptr[i] << std::endl;
         }
      }
   }

   auto syn_start = std::chrono::high_resolution_clock::now();
   for (size_t i = 0; i < ITERS; ++i)
   {
      synthetic_forward(input);
   }
   auto syn_end = std::chrono::high_resolution_clock::now();
   auto syn_dur = std::chrono::duration_cast<std::chrono::microseconds>(syn_end - syn_start).count();

   auto jit_start = std::chrono::high_resolution_clock::now();
   for (size_t i = 0; i < ITERS; ++i)
   {
      pytorch_jit_forward({input});
   }
   auto jit_end = std::chrono::high_resolution_clock::now();
   auto jit_dur = std::chrono::duration_cast<std::chrono::microseconds>(jit_end - jit_start).count();

   std::cerr << "Number of iterations(calls): " << ITERS << std::endl;
   std::cout << "Ultra run: " << syn_dur << " us" << std::endl;
   std::cout << "PyTorch jit run: " << jit_dur << " us" << std::endl;
   std::cout << "Speed up: " << jit_dur / (0.0 + syn_dur) << "X" << std::endl;
}
