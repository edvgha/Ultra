#include "forward.h"
#include <torch/csrc/jit/serialization/import.h>
#include <chrono>

#define ITERS 5000

c10::IValue pytorch_jit_forward(const std::vector<c10::IValue>& inputs)
{
   auto module = torch::jit::load("../Demo_DeepAndWide/DeepAndWide.pt");
   return module.forward(inputs);
}

int main()
{
   long long batch_size = 64;
   long long embedding_size = 32;
   long long num_features = 50;

   Tensor gad_emb_packed_1 = rand({batch_size, 1, embedding_size});
   Tensor guser_emb_1 = rand({batch_size, 1, embedding_size});
   Tensor gwide_1 = rand({batch_size, num_features});
   // Two times to execute 'else' branch too
   {
      // Run Generated 'forward' function
      Tensor synthetic_out;
      std::tie(synthetic_out) = synthetic_forward(gad_emb_packed_1, guser_emb_1, gwide_1);
      auto actual_ptr = synthetic_out.data_ptr<float>();
      // Run PyTorch JIT 
      IValue pytorch_jit_out = pytorch_jit_forward({gad_emb_packed_1, guser_emb_1, gwide_1});
      auto expected = pytorch_jit_out.toTuple()->elements()[0].toTensor();
      auto expected_ptr = expected.data_ptr<float>();
      

      std::stringstream msg;
      msg << "Number of elements of expected and actual outputs don't match\n";
      TORCH_CHECK(expected.numel() == synthetic_out.numel(),  msg.str());
      for (size_t i = 0; i < expected.numel(); ++i) 
      {
         if (std::abs(expected_ptr[i] - actual_ptr[i]) >= 0.0001) 
         {
            std::stringstream msg;
            msg << "Correctness check failed: " << i 
                << " expected: " << expected_ptr[i] 
                << " actual: " << actual_ptr[i] << '\n';
            TORCH_CHECK(false, msg.str());
         }
      }
   }
   {
      // Run Generated 'forward' function
      Tensor synthetic_out;
      std::tie(synthetic_out) = synthetic_forward(gad_emb_packed_1, guser_emb_1, gwide_1);
      auto actual_ptr = synthetic_out.data_ptr<float>();
      // Run PyTorch JIT 
      IValue pytorch_jit_out = pytorch_jit_forward({gad_emb_packed_1, guser_emb_1, gwide_1});
      auto expected = pytorch_jit_out.toTuple()->elements()[0].toTensor();
      auto expected_ptr = expected.data_ptr<float>();
      

      std::stringstream msg;
      msg << "Number of elements of expected and actual outputs don't match\n";
      TORCH_CHECK(expected.numel() == synthetic_out.numel(),  msg.str());
      for (size_t i = 0; i < expected.numel(); ++i) 
      {
         if (std::abs(expected_ptr[i] - actual_ptr[i]) >= 0.0001) 
         {
            std::stringstream msg;
            msg << "Correctness check failed: " << i 
                << " expected: " << expected_ptr[i] 
                << " actual: " << actual_ptr[i] << '\n';
            TORCH_CHECK(false, msg.str());
         }
      }
   }
   // Measure runtime Generated vs PyTorch JIT
   auto syn_start = std::chrono::high_resolution_clock::now();
   for (size_t i = 0; i < ITERS; ++i)
   {
      synthetic_forward(gad_emb_packed_1, guser_emb_1, gwide_1);
   }
   auto syn_end = std::chrono::high_resolution_clock::now();
   auto syn_dur = std::chrono::duration_cast<std::chrono::microseconds>(syn_end - syn_start).count();

   auto jit_start = std::chrono::high_resolution_clock::now();
   for (size_t i = 0; i < ITERS; ++i)
   {
      pytorch_jit_forward({gad_emb_packed_1, guser_emb_1, gwide_1});
   }
   auto jit_end = std::chrono::high_resolution_clock::now();
   auto jit_dur = std::chrono::duration_cast<std::chrono::microseconds>(jit_end - jit_start).count();

   std::cerr << "Number of iterations(calls): " << ITERS << std::endl;
   std::cout << "Ultra run: " << syn_dur << " us" << std::endl;
   std::cout << "PyTorch jit run: " << jit_dur << " us" << std::endl;
   std::cout << "Speed up: " << jit_dur / (0.0 + syn_dur) << "X" << std::endl;
}
