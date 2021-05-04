#pragma once

#include <ATen/ATen.h>

namespace O 
{

template<typename scalar_t>
void ultra_batch_norm(at::Tensor& output, const at::Tensor& input, const at::Tensor& weight, 
                      const at::Tensor& bias, const at::Tensor& save_mean, const at::Tensor& save_invstd,
                      const at::Tensor& running_mean, const at::Tensor& running_var, bool train, double eps);
} // namespace O
