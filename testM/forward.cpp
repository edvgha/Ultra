#include "forward.h"
#include <ATen/core/grad_mode.h>

bool first_time = true;

Tensor g12;

bool g10;

Tensor g9;

Tensor g8;

Tensor g11;

bool g7;

const int g3 = 1;

const int g4 = 0;

const int g5 = 2;

Tensor g13;

const int g6 = 4;



Tensor synthetic_forward (Tensor& gx_1, int& gy_1) 
{
	NoGradGuard no_grad;

  if (first_time) {
    g7 = gy_1 > g4;
    if (g7) {
      g9 = native::add (gx_1, g5, g3);
      g8 = g9;
    } else {
      g10 = gy_1 < g4;
      if (g10) {
        g12 = native::mul (gx_1, g5);
        g11 = g12;
      } else {
        g13 = sub (gx_1, g6, g3);
        g11 = g13;
      }
      g8 = g11;
    }
    first_time = false;
  } else {
    g7 = gy_1 > g4;
    if (g7) {
      g9 = native::add (gx_1, g5, g3);
      g8 = g9;
    } else {
      g10 = gy_1 < g4;
      if (g10) {
        g12 = native::mul (gx_1, g5);
        g11 = g12;
      } else {
        g13 = sub (gx_1, g6, g3);
        g11 = g13;
      }
      g8 = g11;
    }
  }
  return g8;
}
