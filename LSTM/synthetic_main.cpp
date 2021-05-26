#include "forward.h"
#include <torch/csrc/jit/serialization/import.h>
#include <chrono>

#define ITERS 5000

c10::IValue pytorch_jit_forward(const Tensor& input, 
                                const std::tuple<Tensor, Tensor>& x, 
                                const Tensor& w_ih,
                                const Tensor& w_hh, 
                                const Tensor& b_ih, 
                                const Tensor& b_hh)
{
   auto module = torch::jit::load("../LSTM/LSTM.pt");
   return module.forward({input, x, w_ih, w_hh, b_ih, b_hh});
}

int main()
{
   long long seq_length = 3;
   Tensor input = rand({seq_length, 3, 10});
   Tensor hx = rand({3, 20});
   Tensor cx = rand({3, 20});
   std::tuple<Tensor, Tensor> x = std::make_tuple(hx, cx);
   Tensor w_ih = rand({80, 10});
   Tensor w_hh = rand({80, 20});
   Tensor b_ih = rand({80});
   Tensor b_hh = rand({80});
   // Two times to execute 'else' branch too
   {
      // Run Generated 'forward' function
      Tensor syn_0, syn_1;
      std::tie(syn_0, syn_1) = synthetic_forward(input, x, w_ih, w_hh, b_ih, b_hh);
      auto actual_ptr_0 = syn_0.data_ptr<float>();
      auto actual_ptr_1 = syn_1.data_ptr<float>();
      // Run PyTorch JIT
      IValue pytorch_jit_out = pytorch_jit_forward(input, {hx, cx}, w_ih, w_hh, b_ih, b_hh);
      auto expected_elements = pytorch_jit_out.toTuple()->elements();
      TORCH_CHECK(expected_elements.size() == 2);
      auto exp_0 = expected_elements[0].toTensor();
      auto exp_1 = expected_elements[1].toTensor();
      auto expected_ptr_0 = exp_0.data_ptr<float>();
      auto expected_ptr_1 = exp_1.data_ptr<float>();
      
      std::stringstream msg;
      msg << "Number of elements of expected and actual outputs don't match\n";
      TORCH_CHECK(exp_0.numel() == syn_0.numel(),  msg.str());
      TORCH_CHECK(exp_1.numel() == syn_1.numel(), msg.str());
      for (size_t i = 0; i < exp_0.numel(); ++i) 
      {
         if (std::abs(expected_ptr_0[i] - actual_ptr_0[i]) >= 0.0001) 
         {
            std::stringstream msg;
            msg << "Correctness check failed: " << i 
                << " expected: " << expected_ptr_0[i] 
                << " actual: " << actual_ptr_0[i] << '\n';
            TORCH_CHECK(false, msg.str());
         }
      }
      for (size_t i = 0; i < exp_1.numel(); ++i) 
      {
         if (std::abs(expected_ptr_1[i] - actual_ptr_1[i]) >= 0.0001) 
         {
            std::stringstream msg;
            msg << "Correctness check failed: " << i 
                << " expected: " << expected_ptr_1[i] 
                << " actual: " << actual_ptr_1[i] << '\n';
            TORCH_CHECK(false, msg.str());
         }
      }
   }
   {
      // Run Generated 'forward' function
      Tensor syn_0, syn_1;
      std::tie(syn_0, syn_1) = synthetic_forward(input, x, w_ih, w_hh, b_ih, b_hh);
      auto actual_ptr_0 = syn_0.data_ptr<float>();
      auto actual_ptr_1 = syn_1.data_ptr<float>();
      // Run PyTorch JIT
      IValue pytorch_jit_out = pytorch_jit_forward(input, {hx, cx}, w_ih, w_hh, b_ih, b_hh);
      auto expected_elements = pytorch_jit_out.toTuple()->elements();
      TORCH_CHECK(expected_elements.size() == 2);
      auto exp_0 = expected_elements[0].toTensor();
      auto exp_1 = expected_elements[1].toTensor();
      auto expected_ptr_0 = exp_0.data_ptr<float>();
      auto expected_ptr_1 = exp_1.data_ptr<float>();
      
      std::stringstream msg;
      msg << "Number of elements of expected and actual outputs don't match\n";
      TORCH_CHECK(exp_0.numel() == syn_0.numel(),  msg.str());
      TORCH_CHECK(exp_1.numel() == syn_1.numel(), msg.str());
      for (size_t i = 0; i < exp_0.numel(); ++i) 
      {
         if (std::abs(expected_ptr_0[i] - actual_ptr_0[i]) >= 0.0001) 
         {
            std::stringstream msg;
            msg << "Correctness check failed: " << i 
                << " expected: " << expected_ptr_0[i] 
                << " actual: " << actual_ptr_0[i] << '\n';
            TORCH_CHECK(false, msg.str());
         }
      }
      for (size_t i = 0; i < exp_1.numel(); ++i) 
      {
         if (std::abs(expected_ptr_1[i] - actual_ptr_1[i]) >= 0.0001) 
         {
            std::stringstream msg;
            msg << "Correctness check failed: " << i 
                << " expected: " << expected_ptr_1[i] 
                << " actual: " << actual_ptr_1[i] << '\n';
            TORCH_CHECK(false, msg.str());
         }
      }
   }
   
   
   // Measure runtime Generated vs PyTorch JIT
   auto syn_start = std::chrono::high_resolution_clock::now();
   for (size_t i = 0; i < ITERS; ++i)
   {
      
      synthetic_forward(input, x, w_ih, w_hh, b_ih, b_hh);
   }
   auto syn_end = std::chrono::high_resolution_clock::now();
   auto syn_dur = std::chrono::duration_cast<std::chrono::microseconds>(syn_end - syn_start).count();

   auto jit_start = std::chrono::high_resolution_clock::now();
   for (size_t i = 0; i < ITERS; ++i)
   {
      pytorch_jit_forward(input, {hx, cx}, w_ih, w_hh, b_ih, b_hh);
   }
   auto jit_end = std::chrono::high_resolution_clock::now();
   auto jit_dur = std::chrono::duration_cast<std::chrono::microseconds>(jit_end - jit_start).count();

   std::cerr << "Number of iterations(calls): " << ITERS << std::endl;
   std::cout << "Ultra run: " << syn_dur << " us" << std::endl;
   std::cout << "PyTorch jit run: " << jit_dur << " us" << std::endl;
   std::cout << "Speed up: " << jit_dur / (0.0 + syn_dur) << "X" << std::endl;
}
