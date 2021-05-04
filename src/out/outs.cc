#include "outs.h"
#include <ATen/native/TensorIterator.h>
#include <ATen/cpu/vec256/vec256.h>
#include <iostream>

namespace O 
{

using namespace at;

namespace 
{

using namespace at::vec256;

template <typename scalar_t>
static TensorAccessor<scalar_t, 1> conditional_accessor_1d(const Tensor& t) {
  if (! t.defined()) {
    return TensorAccessor<scalar_t, 1>(nullptr, nullptr, nullptr);
  }
  return t.accessor<scalar_t, 1>();
}

template<typename scalar_t>
void batch_norm_cpu_inference_channels_last(Tensor& output, const Tensor& input,
    const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& mean, const Tensor& variance, double eps) {

  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  scalar_t* output_data = output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.data_ptr<scalar_t>();

  Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  scalar_t* alpha_data = alpha.data_ptr<scalar_t>();
  scalar_t* beta_data = beta.data_ptr<scalar_t>();

  batch_norm_cpu_inference_collect_linear_and_constant_terms<scalar_t>(
      alpha_data, beta_data, n_channel, weight, bias, mean, variance, eps);

  // Apply the linear terms to the input,
  // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
  // No need to use parallel_for as this function is supposed to be
  // memory-limited.
  // Keep the loop structure simple to make sure compiler vectorization kicks in.
  if (n_channel != 1) {
    for (int64_t n = 0; n < n_batch; ++n) {
      for (int64_t i = 0; i < image_size; ++i) {
        for (int64_t c = 0; c < n_channel; ++c) {
          // Keep all the offset calculation within the inner loop for
          // simplicity. Compilers are very good at hoisting the common part
          // outside.
          int64_t offset = n * image_size * n_channel + i * n_channel + c;
          output_data[offset] = input_data[offset] * alpha_data[c] + beta_data[c];
        }
      }
    }
  } else {
    // n_channel == 1
    for (int64_t n = 0; n < n_batch; ++n) {
      for (int64_t i = 0; i < image_size; ++i) {
        int64_t offset = n * image_size + i;
        output_data[offset] = input_data[offset] * alpha_data[0] + beta_data[0];
      }
    }
  }
}

template<typename scalar_t>
void batch_norm_cpu_inference_collect_linear_and_constant_terms(
    TensorAccessor<scalar_t, 1> alpha, TensorAccessor<scalar_t, 1> beta, int64_t n_channel,
    const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& mean, const Tensor& variance, double eps) {

  const scalar_t* weight_data = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
  const scalar_t* bias_data = bias.defined() ? bias.data_ptr<scalar_t>() : nullptr;
  auto mean_data = mean.accessor<scalar_t, 1>();
  auto var_data = variance.accessor<scalar_t, 1>();

  /// Collect the linear and constant terms regarding the input.
  /// output(n, c, h, w)
  ///     = (input(n, c, h, w) - mean(c)) / sqrt(var(c) + eps) * weight(c)
  ///         + bias(c)
  ///     = input(n, c, h, w) * inv_var(c) * weight(c)
  ///         - mean(c) * inv_var(c) * weight(c) + bias(c),
  /// where inv_var(c) = 1 / sqrt(var(c) + eps).
  /// So the linear term, alpha(c) = inv_var(c) * weight(c),
  ///   the constant term beta(c) = bias(c) - mean(c) * inv_var(c) * weight(c)
  /// Note that this is only a good idea if (input_size >> c), in degenerate
  /// cases where image_size == 1 && batch_size == 1, it is slow.
  for (int64_t c = 0; c < n_channel; c++) {
    scalar_t inv_var = 1 / std::sqrt(var_data[c] + static_cast<scalar_t>(eps));
    scalar_t weight_v = weight_data ? weight_data[c] : 1;
    scalar_t bias_v = bias_data ? bias_data[c] : 0;
    alpha[c] = inv_var * weight_v;
    beta[c] = bias_v - mean_data[c] * alpha[c];
  }
}

template<typename scalar_t>
void batch_norm_cpu_inference_contiguous_impl(Tensor& output,
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& mean, const Tensor& variance, double eps) {

  using Vec = Vec256<scalar_t>;
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto alpha_data = alpha.accessor<scalar_t, 1>();
  auto beta_data = beta.accessor<scalar_t, 1>();

  batch_norm_cpu_inference_collect_linear_and_constant_terms<scalar_t>(
     alpha_data, beta_data, n_channel, weight, bias, mean, variance, eps);

  scalar_t* output_data = output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.data_ptr<scalar_t>();

  // Apply the linear terms to the input,
  // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
  // No need to use parallel_for as this function is supposed to be
  // memory-limited.
  if (image_size != 1) {
    const int64_t n_offset = n_channel * image_size;
    const int64_t loop_size = image_size - (image_size % Vec::size());
    for (int64_t n = 0; n < n_batch; n++) {
      for (int64_t c = 0; c < n_channel; c++) {
        const Vec alpha_vec(alpha_data[c]);
        const Vec beta_vec(beta_data[c]);
        int64_t offset = n * n_offset + c * image_size;
        int64_t d = 0;
        for (; d < loop_size; d += Vec::size()) {
          Vec data_vec = Vec::loadu(input_data + offset + d);
          Vec output_vec = data_vec * alpha_vec + beta_vec;
          output_vec.store(output_data + offset + d);
        }
        if (image_size - d > 0) {
          Vec data_vec = Vec::loadu(input_data + offset + d, image_size - d);
          Vec output_vec = data_vec * alpha_vec + beta_vec;
          output_vec.store(output_data + offset + d, image_size - d);
        }
      }
    }
  } else {
    // image_size == 1
    for (int64_t n = 0; n < n_batch; ++n) {
      for (int64_t c = 0; c < n_channel; ++c) {
        int64_t offset = n * n_channel + c;
        output_data[offset] = input_data[offset] * alpha_data[c] + beta_data[c];
      }
    }
  }
}

} // namespace

template <typename scalar_t>
void ultra_batch_norm(Tensor& output, const Tensor& input, const Tensor& weight, 
                      const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd,
                      const Tensor& running_mean, const Tensor& running_var, bool train, double eps)
{
  if (output.is_contiguous()
      && (!weight.defined() || weight.is_contiguous())
      && (!bias.defined() || bias.is_contiguous())
      && running_mean.is_contiguous()
      && running_var.is_contiguous()) {

    batch_norm_cpu_inference_contiguous_impl<scalar_t>(output, input, weight,
        bias, running_mean, running_var, eps);
    return ;
  }

  if (output.is_contiguous(at::MemoryFormat::ChannelsLast)
      && (!weight.defined() || weight.is_contiguous())
      && (!bias.defined() || bias.is_contiguous())
      && running_mean.is_contiguous()
      && running_var.is_contiguous()) {

    batch_norm_cpu_inference_channels_last<scalar_t>(
        output, input, weight, bias, running_mean, running_var, eps);
    return ;
  }

  int64_t n_input = input.size(1);

  auto save_mean_a = conditional_accessor_1d<scalar_t>(save_mean);
  auto save_invstd_a = conditional_accessor_1d<scalar_t>(save_invstd);

  auto running_mean_a = conditional_accessor_1d<scalar_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<scalar_t>(running_var);

  parallel_for(0, n_input, 1, [&](int64_t b_begin, int64_t b_end) {
    for (int64_t f = b_begin; f < b_end; ++f) {
      Tensor in = input.select(1, f);
      Tensor out = output.select(1, f);

      scalar_t mean, invstd;
      if (train) {
        mean = save_mean_a[f];
        invstd = save_invstd_a[f];
      } else {
        mean = running_mean_a[f];
        invstd = 1 / std::sqrt(running_var_a[f] + eps);
      }

      // compute output
      scalar_t w = weight.defined() ? weight.data_ptr<scalar_t>()[f * weight.stride(0)] : 1;
      scalar_t b = bias.defined() ? bias.data_ptr<scalar_t>()[f * bias.stride(0)] : 0;

      auto iter = TensorIterator::unary_op(out, in);
      cpu_serial_kernel(iter, [=](const scalar_t i) -> scalar_t {
        return ((i - mean) * invstd) * w + b;
      });
    }
  });
}

} // namespace O