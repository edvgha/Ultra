#include "forward.h"
#include <ATen/core/grad_mode.h>

bool first_time = true;

Tensor g38;

std::tuple<Tensor, Tensor> ghidden0_2;

Tensor gcellgate0_1;

Tensor gforgetgate0_1;

const bool g7 = 1;

Tensor goutgate_1;

Tensor ghy_1;

Tensor gforgetgate_1;

const int g9 = 1;

std::vector<Tensor>  g26;

Tensor goutgate0_1;

Tensor ggates_1;

Tensor g24;

Tensor g21;

Tensor gingate0_1;

Tensor g19;

Tensor ghx_1;

std::tuple<Tensor, Tensor>  ghidden0_5;

Tensor g16;

Tensor g35;

Tensor gingate_1;

Tensor g22;

std::vector<Tensor>  ginputs_1;

Tensor gcx_1;

Tensor gcy_1;

Tensor g20;

Tensor g23;

Tensor g36;

std::tuple<Tensor, Tensor>  ghidden0;

Tensor gcellgate_1;

const int g8 = 0;

const int g10 = 4;



std::tuple<Tensor, Tensor>  synthetic_forward (Tensor& ginput_1, std::tuple<Tensor, Tensor> & ghidden_1, Tensor& gwih_1, Tensor& gwhh_1, Tensor& gbih_1, Tensor& gbhh_1) 
{
	NoGradGuard no_grad;

  if (first_time) {
    ginputs_1 = native::unbind (ginput_1, g8);
    int g12 = ginputs_1.size();
    ghidden0 = ghidden_1;
    bool prim_loop_condition_1 = g7;
    int gseq_idx_1 = 0;
    while (prim_loop_condition_1 and gseq_idx_1 < g12) {
      ghidden0_5 = ghidden0;
      g16 = ginputs_1[gseq_idx_1];
      std::tie(ghx_1, gcx_1) = ghidden0_5;
      g19 = native::t (gwih_1);
      g20 = native::mm_cpu (g16, g19);
      g21 = native::t (gwhh_1);
      g22 = native::mm_cpu (ghx_1, g21);
      g23 = native::add (g20, g22, g9);
      g24 = native::add (g23, gbih_1, g9);
      ggates_1 = native::add (g24, gbhh_1, g9);
      g26 = native::chunk (ggates_1, g10, g9);
      gingate_1 = g26[0];
      gforgetgate_1 = g26[1];
      gcellgate_1 = g26[2];
      goutgate_1 = g26[3];
      gingate0_1 = native::sigmoid (gingate_1);
      gforgetgate0_1 = native::sigmoid (gforgetgate_1);
      gcellgate0_1 = native::tanh (gcellgate_1);
      goutgate0_1 = native::sigmoid (goutgate_1);
      g35 = native::mul (gforgetgate0_1, gcx_1);
      g36 = native::mul (gingate0_1, gcellgate0_1);
      gcy_1 = native::add (g35, g36, g9);
      g38 = native::tanh (gcy_1);
      ghy_1 = native::mul (goutgate0_1, g38);
      ghidden0_2 = {ghy_1, gcy_1};
      ghidden0 = ghidden0_2;
      prim_loop_condition_1 = g7;
      gseq_idx_1 += 1;
    }
    first_time = false;
  } else {
    ginputs_1 = native::unbind (ginput_1, g8);
    int g12 = ginputs_1.size();
    ghidden0 = ghidden_1;
    bool prim_loop_condition_2 = g7;
    int gseq_idx_1 = 0;
    while (prim_loop_condition_2 and gseq_idx_1 < g12) {
      ghidden0_5 = ghidden0;
      g16 = ginputs_1[gseq_idx_1];
      std::tie(ghx_1, gcx_1) = ghidden0_5;
      g19 = native::t (gwih_1);
      native::mm_cpu_out (g20, g16, g19);
      g21 = native::t (gwhh_1);
      native::mm_cpu_out (g22, ghx_1, g21);
      native::add_out (g23, g20, g22, g9);
      native::add_out (g24, g23, gbih_1, g9);
      native::add_out (ggates_1, g24, gbhh_1, g9);
      g26 = native::chunk (ggates_1, g10, g9);
      gingate_1 = g26[0];
      gforgetgate_1 = g26[1];
      gcellgate_1 = g26[2];
      goutgate_1 = g26[3];
      native::sigmoid_out (gingate0_1, gingate_1);
      native::sigmoid_out (gforgetgate0_1, gforgetgate_1);
      native::tanh_out (gcellgate0_1, gcellgate_1);
      native::sigmoid_out (goutgate0_1, goutgate_1);
      native::mul_out (g35, gforgetgate0_1, gcx_1);
      native::mul_out (g36, gingate0_1, gcellgate0_1);
      native::add_out (gcy_1, g35, g36, g9);
      native::tanh_out (g38, gcy_1);
      native::mul_out (ghy_1, goutgate0_1, g38);
      ghidden0_2 = {ghy_1, gcy_1};
      ghidden0 = ghidden0_2;
      prim_loop_condition_2 = g7;
      gseq_idx_1 += 1;
    }
  }
  return ghidden0;
}
