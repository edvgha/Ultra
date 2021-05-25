#include "../scripts.h"
#include <ultra.h>
#include <vector>

namespace {

void graph_executor(const std::string& name, 
                    const std::vector<c10::IValue>& inputs, 
                    const std::string& s)
{
    ultra::fs::path p = ultra::fs::current_path();
    p /= s;
    std::stringstream ss;
    ss << "mkdir -p " << p.c_str();
    std::system(ss.str().c_str());
    p /= "test_gout";

    torch::jit::script::Module module("module");
    module.define(name);

    auto jit_res = module.forward(inputs);

    if (jit_res.isTuple())
    {
        auto r = jit_res.toTuple()->elements();
        for (size_t i = 0; i < r.size(); ++i)
        {
            std::ofstream(p) << r[i] << '\n';
        }
    }
    else if (jit_res.isTensor()) 
    {
        auto r = jit_res.toTensor();
        std::ofstream(p) << r << '\n';
    }
    else 
    {
        TORCH_CHECK(jit_res.isList(), "Expected to be List.");
        auto r = jit_res.toTensorVector();
        for (size_t i = 0; i < r.size(); ++i)
        {
            std::ofstream(p) << r[i] << '\n';
        }
    }
}

void generate(const std::string& name, const std::string& s)
{
    torch::jit::script::Module module("module");
    module.define(name);
    ultra::Ultra u(module);
    ultra::fs::path p = ultra::fs::current_path();
    p /= s;
    u.buildLibrary(p);
}

void generate_main(const std::string& name, 
                   const std::vector<std::pair<std::string, std::string>>& inputs, 
                   const std::string& s)
{
    ultra::fs::path p = ultra::fs::current_path();
    p /= s;
    p /= "test_main.cc";
    std::ostringstream oss;
    oss << "#include \"forward.h\"\n"
        << "#include <torch/csrc/jit/serialization/import.h>\n\n"
        << "int main() {\n";
    for (auto i = 0; i < inputs.size(); ++i) 
    {
        oss << '\t' << inputs[i].first << " in" << i << " = " << inputs[i].second << ";\n";
    }
    oss << "\tTensor out = synthetic_forward (";
    for (auto i = 0; i < inputs.size(); ++i) 
    {
        oss << "in" << i;
        if (i < inputs.size() - 1) 
        {
            oss << ",";
        }
    }
    oss << ");\n"
        << "\tstd::cout << out << std::endl;\n"
        << "}\n";
    std::ofstream(p) << oss.str();
}

void test(const std::string& name, 
          const std::vector<c10::IValue>& inputs, 
          const std::vector<std::pair<std::string, std::string>>& inputs_str, 
          const std::string& s)
{
    graph_executor(name, inputs, s);
    generate(name, s);
    generate_main(name, inputs_str, s);
}

} // namespace

int main(int argc, char** argv)
{
    test(ir_for_for, {at::ones({2, 2}), at::ones({2, 2})}, {{"Tensor", "at::ones({2, 2})"}, {"Tensor", "at::ones({2, 2})"}}, "1");
}
