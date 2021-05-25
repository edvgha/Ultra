#include "forward.h"
#include <ATen/core/grad_mode.h>

bool first_time = true;

Tensor g16;

c10::optional<ScalarType> g3 = c10::nullopt;

Tensor gz_1;

const bool g2 = 1;

const int g5 = 1;

const std::array<int64_t, 2> g19 = {2, 2};

Tensor gz_3;

const int g18 = 4;

Tensor oness(
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  return native::ones(size, options);
}


Tensor synthetic_forward (Tensor& gx)
{
        NoGradGuard no_grad;

  // if (first_time) {
    //gz_1 = oness (g19, g3, g3, g3, g3);
    //Tensor gz = gz_1;
    /*
    bool prim_loop_condition_1 = g2;
    int g13 = 0;
    while (prim_loop_condition_1 and g13 < g18) {
      gz_6 = gz;
      g16 = native::randn (g19, g3, g3, g3, g3);
      gz_3 = native::add_ (gz_6, g16, g5);
      gz = gz_3;
      prim_loop_conditions_1 = g2;
    }
    first_time = false;
  } else {
    native::ones_out (gz_1, g19, g3, g3, g3, g3);
    Tensor gz = gz_1;
    bool prim_loop_condition_2 = g2;
    int g13 = 0;
    while (prim_loop_condition_2 and g13 < g18) {
      gz_6 = gz;
      native::randn_out (g16, g19, g3, g3, g3, g3);
      gz_3 = native::add_ (gz_6, g16, g5);
      gz = gz_3;
      prim_loop_conditions_2 = g2;
    }
    */
  // } 
  return gz_3;
}
