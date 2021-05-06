#include "forward.h"
#include <ATen/core/grad_mode.h>

bool first_time = true;

const Tensor g3 = {};

Tensor g10;

Tensor g8;

Tensor g7;

const bool g4 = 1;

const int g5 = 2;

const int g6 = 1;

const std::array<int64_t, 1> g11 = {1};



Tensor synthetic_forward (Tensor& gx_1, bool& gy_1) 
{
	NoGradGuard no_grad;
  /*
  if (first_time) {
    if (gy_1) {
      g8 = native::add (gx_1, g5, g6);
      g7 = g8;
    } else {
      g10 = sum (gx_1, g11, g4, g3);
      g7 = g10;
    }
    first_time = false;
  } else {
    if (gy_1) {
      g8 = native::add (gx_1, g5, g6);
      g7 = g8;
    } else {
      g10 = sum (gx_1, g11, g4, g3);
      g7 = g10;
    }
  }*/
  return g7;
}
