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
        auto r = jit_res.toTuple()->elements();
        for (size_t i = 0; i < r.size(); ++i)
        {
            std::cout << r[i] << std::endl;
        }
    }
    else if (jit_res.isTensor()) 
    {
        auto r = jit_res.toTensor();
        std::cout << r << std::endl;
    }
    else 
    {
        TORCH_CHECK(jit_res.isList(), "Expected to be List.");
        auto r = jit_res.toTensorVector();
        for (size_t i = 0; i < r.size(); ++i)
        {
            std::cout << r[i] << std::endl;
        }
    }
}
}

int main(int argc, char** argv)
{
    test_functional(ir_if, {at::ones({2, 2}), true});
    test_functional(ir_if, {at::ones({2, 2}), false});
    test_functional(ir_if_1, {at::ones({2, 2}), true});
    test_functional(ir_if_1, {at::ones({2, 2}), false});
    return 0;
}