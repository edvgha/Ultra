#include "scripts.h"
#include <ultra.h>
#include <vector>

namespace {

void test_functional(const std::string& name, const std::vector<c10::IValue>& inputs)
{ 
    torch::jit::script::Module module("module");
    module.define(name);

    auto jit_res = module.forward(inputs);

    if (jit_res.isTuple())
    {
        std::cout << "TUPLE" << std::endl;
        auto r = jit_res.toTuple()->elements();
        for (size_t i = 0; i < r.size(); ++i)
        {
            std::cout << r[i] << std::endl;
        }
    }
    else
    {
        std::cout << "LIST" << std::endl;
        TORCH_CHECK(jit_res.isList(), "Expected to be List.");
        auto r = jit_res.toTensorVector();
        for (size_t i = 0; i < r.size(); ++i)
        {
            std::cout << r[i] << std::endl;
        }
    }
}

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
    // + test(aten_sum_1_true);
    // + test(ir_if);
    // + test(ir_if_1);
    // + test(ir_if_2);
    // + test(ir_for);
    // + test(ir_for_for);
    // + test(ir_for_for_if);
    test(list_construct_script);
    // + test(tuple_construct_script);
    // 3 + test(add_script);
    // 4 + test(reshape_script_1);
    // 5 + test(reshape_script_2);
    // + test(flatten_script_1);
    // + test(flatten_script_2);
    // + test(aten_sum);
    // + test(aten_sum_0);
    // + test(aten_sum_1);
    // + test(aten_sum_0_true);
    // + test(aten_sum_1_true);
    // + test(ir_lstm_cell);
}

void list_construct_test()
{
    test_functional(list_construct_script, {at::randn({3, 3}), at::randn({3, 3})});
}

void tuple_construct_test()
{
    test_functional(tuple_construct_script, {at::randn({3, 3}), at::randn({3, 3})});
}

}

int main(int argc, char** argv)
{
    unit_tests();
    list_construct_test();
    tuple_construct_test();
    return 0;
}
