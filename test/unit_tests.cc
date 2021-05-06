#include "scripts.h"
#include "scripts_out.h"
#include <ultra.h>
#include <vector>

namespace {

std::string modifyForTest(std::string&& s)
{
    s.erase(std::remove_if(s.begin(), s.end(),
                              [](unsigned char x){return std::isspace(x) or (x == '\"');}),
                              s.end());
    return s;
}

void test(const std::string& name, const std::string& name_out)
{
    torch::jit::script::Module module("module");
    module.define(name);
    ultra::Ultra u(module);
    ultra::fs::path p;
    std::optional<std::string> r = u.buildLibrary(p);
    TORCH_CHECK(r.has_value());
    TORCH_CHECK(modifyForTest(std::move(r.value())) == name_out)
}

void unit_tests()
{    
    test(ir_if, ir_if_out);
    // test(ir_if_1, ir_if_1_out);
    // test(ir_if_2);
    // test(ir_for);
    // test(ir_for_for);
    // test(ir_for_for_if);
    // test(list_construct_script);
    // test(tuple_construct_script);
    // test(add_script);
    // test(reshape_script_1);
    // test(reshape_script_2);
    // test(flatten_script_1);
    // test(flatten_script_2);
    // test(aten_sum);
    // test(aten_sum_0);
    // test(aten_sum_1);
    // test(aten_sum_0_true);
    // test(aten_sum_1_true);
}

} // namespace

int main(int argc, char** argv)
{
    unit_tests();
    return 0;
}
