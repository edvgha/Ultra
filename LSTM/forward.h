#include <ATen/ATen.h>
#include <vector>
#include <tuple>

using namespace at;



std::tuple<Tensor, Tensor>  synthetic_forward (Tensor& ginput_1, std::tuple<Tensor, Tensor> & ghidden_1, Tensor& gwih_1, Tensor& gwhh_1, Tensor& gbih_1, Tensor& gbhh_1);
