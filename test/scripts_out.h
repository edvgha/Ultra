#pragma once 

const auto ir_if_out = "#includeforward.h#include<ATen/core/grad_mode.h>boolfirst_time=true;Tensorg5;Tensorg4;constintg3=2;Tensorsynthetic_forward(Tensor&gx_1,bool&gy_1){NoGradGuardno_grad;if(first_time){if(gy_1){g5=native::mul(gx_1,g3);g4=g5;}else{g4=gx_1;}first_time=false;}else{if(gy_1){g5=native::mul(gx_1,g3);g4=g5;}else{g4=gx_1;}}returng4;}";

const auto ir_if_1_out = "";

const auto ir_if_2_out = "";

const auto ir_for_out = "";

const auto ir_for_for_out = "";

const auto ir_for_for_if_out = "";

const auto list_construct_script_out = "";

const auto list_unpack_script_out = "";

const auto tuple_construct_script_out = "";

const auto add_script_out = "";

const auto reshape_script_1_out = "";

const auto reshape_script_2_out = "";

const auto flatten_script_1_out = "";

const auto flatten_script_2_out = "";

const auto aten_sum_out = "";

const auto aten_sum_0_out = "";

const auto aten_sum_1_out = "";

const auto aten_sum_0_true_out = "";

const auto aten_sum_1_true_out = "";
