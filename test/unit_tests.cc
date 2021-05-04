#include "scripts.h"
#include <ultra.h>
#include <vector>

namespace {

void test(const std::string& name)
{
    torch::jit::script::Module module("module");
    module.define(name);
    ultra::Ultra u(module);
    ultra::fs::path p;
    u.buildLibrary(p);
}

void unit_tests()
{
    test(aten_sum_1_true);
    test(ir_if);
    test(ir_if_1);
    test(ir_if_2);
    test(ir_for);
    test(ir_for_for);
    test(ir_for_for_if);
    test(list_construct_script);
    test(tuple_construct_script);
    test(add_script);
    test(reshape_script_1);
    test(reshape_script_2);
    test(flatten_script_1);
    test(flatten_script_2);
    test(aten_sum);
    test(aten_sum_0);
    test(aten_sum_1);
    test(aten_sum_0_true);
    test(aten_sum_1_true);
}

} // namespace

int main(int argc, char** argv)
{
    unit_tests();
    return 0;
}
