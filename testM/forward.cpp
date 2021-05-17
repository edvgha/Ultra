#include "forward.h"
#include <ATen/core/grad_mode.h>

bool first_time = true;

Tensor g16;

Tensor g15;

const int g6 = 1;

Tensor g9;

Tensor g14;

bool g13;

const int g4 = 0;

Tensor g10;

Tensor g12;

const int g5 = 2;

Tensor g11;

bool g8;

const int g7 = 4;



Tensor synthetic_forward (Tensor& gx_1, bool& gy_1, int& gz_1) 
{
	NoGradGuard no_grad;

  if (first_time) {
    g8 = gz_1 >  g4;
    if (g8) {
      if (gy_1) {
        g11 = native::add (gx_1, g5, g6);
        g10 = g11;
      } else {
        g12 = native::add (gx_1, g6, g6);
        g10 = g12;
      }
      g9 = g10;
    } else {
      g13 = (gz_1 < g4);
      if (g13) {
        g15 = native::mul (gx_1, g5);
        g14 = g15;
      } else {
        g16 = native::sub (gx_1, g7, g6);
        g14 = g16;
      }
      g9 = g14;
    }
    first_time = false;
  } else {
    g8 = (gz_1 > g4);
    if (g8) {
      if (gy_1) {
        //native::add_out (g11, gx_1, g5, g6);
        g10 = g11;
      } else {
        //native::add_out (g12, gx_1, g6, g6);
        g10 = g12;
      }
      g9 = g10;
    } else {
      g13 = (gz_1 < g4);
      if (g13) {
        g15 = native::mul (gx_1, g5);
        g14 = g15;
      } else {
        g16 = sub (gx_1, g7, g6);
        g14 = g16;
      }
      g9 = g14;
    }
  }
  return g9;
}
