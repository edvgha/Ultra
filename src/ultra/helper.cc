#include "ultra.h"

#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <exception>
#include <cstdlib>
#include <iostream>

// node -> matches("aten::size(Tensor self) -> int[]")

namespace ultra {

using namespace torch::jit;

std::string Ultra::generateCPPForwardCode() const
{
    // constValueListWithTypes(inputs());
    std::vector<const Node*> groups;
    for (auto n : graph_ -> nodes()) 
    {
        processNode(n, &groups);
    }

    
    processReturns();

    for (auto fg : groups) 
    {
        std::cerr << " *** GROUP ***" << std::endl;
        std::cerr << fg->kind().toQualString() << std::endl;
        std::cerr << *fg->g(attr::Subgraph) << std::endl;
    }
    
    std::ostringstream oss;
    return oss.str();
}

void Ultra::processNode(Node* node, std::vector<const Node*>* groups) const
{
    constValueListWithTypes(node -> outputs());
    if (node -> kind() == prim::PythonOp) 
    {
        auto* pyOp = static_cast<const PythonOp*>(node);
        std::cerr << "pyOp: ^" << pyOp->name() << std::endl; // 252
        // pyOp -> writeScalars
    }
    else if (node -> hasAttribute(attr::Subgraph) and groups)
    {
        std::cerr << "Node kind1: " << node -> kind().toQualString() << std::endl;
        if (node -> numAttributes() > 1 and node -> kind() != prim::DifferentiableGraph) 
        {
            processAttributes(node, true);
        }
        groups -> push_back(node);
    }
    else
    {
        std::cerr << "Node kind2: " << node -> kind().toQualString() << std::endl;
        if (node -> hasAttributes()) 
        {
            processAttributes(node);
        }
    }

    std::cerr << "Node inputs: " << node -> inputs() << std::endl;

    std::string scName = node -> scopeName();

    std::cerr << "scName : " << scName << std::endl;

    for (size_t i = 0; i < node -> blocks().size(); ++i) 
    {
        auto b = node -> blocks()[i];
        constValueListWithTypes(b -> inputs());
        for (auto nested : b -> nodes()) 
        {
            processNode(nested, groups);
        }
        auto blockOutputs = b -> outputs();
        for (auto blOut : blockOutputs) 
        {
            std::cout << "+++ blockOutput: " << blOut -> debugName() << std::endl;
            std::cout << "+++ blockOutput type: " << *blOut -> type() << std::endl;
        }
    }
}

void Ultra::processReturns() const
{
    auto outs = graph_ -> block() -> outputs();
    for (auto out : outs) 
    {
        std::cout << "RetOutput: " << out -> debugName() << std::endl;
        std::cout << "RetOutput type: " << *out -> type() << std::endl;
    }
}

void Ultra::constValueListWithTypes(const c10::ArrayRef<Value*> inputs) const
{
    for (auto i : inputs) 
    {
        std::cout << " >>>> valueRef: " << i -> debugName() << std::endl;
        std::cout << " >>>> valueRef type: " << *i -> type() << std::endl;
    }
}

void Ultra::processAttributes(const Node* node, bool ignore_subgraph) const
{
    auto names = node -> attributeNames();
    for (auto name : names) 
    {
        if (ignore_subgraph and name == attr::Subgraph)
        {
            continue;
        }

        std::cerr << "Attr name: " << name.toUnqualString() << std::endl;
        // attribute value
        processAttributeValue(node, name);
    }
}

void Ultra::processAttributeValue(const Node* node, const Symbol& name) const
{
    switch (node -> kindOf(name)) 
    {
        case AttributeKind::f:
        {
            std::cerr << "Attr val double" << std::endl;
            const double& f = node -> f(name);
            break;
        }
        case AttributeKind::fs:
        {
            std::cerr << "Attr val vector double" << std::endl;
            const std::vector<double>& fs = node -> fs(name);
            break;
        }
        case AttributeKind::i:
        {
            std::cerr << "Attr val int name : " << name << std::endl;
            const int64_t i = node -> i(name);
            break;
        }
        case AttributeKind::is:
        {
            std::cerr << "Attr val vector int" << std::endl;
            const std::vector<int64_t> is = node -> is(name);
            break;
        }
        case AttributeKind::s:
        {
            std::cerr << "Attr val string" << std::endl;
            const std::string s = node -> s(name);
            break;
        }
        case AttributeKind::ss:
        {
            std::cerr << "Attr val vector string" << std::endl;
            const std::vector<std::string> ss = node -> ss(name);
            break;
        }
        case AttributeKind::t:
        {
            std::cerr << "Attr val Tensor" << std::endl;
            const at::Tensor& t = node -> t(name);
            break;
        }
        case AttributeKind::ts:
        {
            std::cerr << "Attr val: " << "[<Tensors>]" << std::endl;
            break;
        }
        case AttributeKind::ival:
        {
            std::cerr << "Attr val IValue" << std::endl;
            const c10::IValue& ival = node -> ival(name);
            break;
        }
        case AttributeKind::g:
        {
            std::cerr << "Attr val: " << "<Graph>" << std::endl;
            break;
        }
        case AttributeKind::gs:
        {
            std::cerr << "Attr val: " << "[<Graphs>]" << std::endl;
            break;
        }
        case AttributeKind::ty:
        {
            std::cerr << "Attr val TypePtr" << std::endl;
            const c10::TypePtr ty = node -> ty(name);
            break;
        }
        case AttributeKind::tys:
        {
            std::cerr << "Attr val vector TypePtr" << std::endl;
            const std::vector<c10::TypePtr> tys = node -> tys(name);
        }
        default : 
        {
            std::cerr << "Unknown attr value" << std::endl;
            break;
        }
  }
}

c10::ArrayRef<Value*> Ultra::inputs() const
{
    return graph_->block()->inputs();
}

void Ultra::setupWorkspace() const
{
    try
    {
        auto w = fs::current_path();
        w /= "w";
        if (fs::is_directory(w) and not fs::remove_all(w))
        {
            std::cerr << "Failed to remove old workspace.\n";
        }
        if (not fs::create_directory(w))
        {
            std::cerr << "Failed to create workspace directory.\n";
        }
    }
    catch(const fs::filesystem_error& e)
    {
        std::cerr << e.what() << '\n';
    }
}

std::string Ultra::declaration()
{
    std::ostringstream oss; 

    const auto& returns = schema_ -> returns();
    TORCH_CHECK(returns.size() == 1, "Currently supported number of outputs is one.");
    const auto& r = returns.at(0);
    if (r.type()->isSubtypeOf(ListType::ofTensors())) 
    {
        oss << "std::vector<Tensor> ";
    }
    else
    {
        TORCH_CHECK(false, "Not supported return type.");
    }

    oss << "synthetic_forward (";
    std::vector<c10::Argument> args = schema_ -> arguments();
    bool seen_kwarg_only = false;
    for(size_t i = 0; i < args.size(); ++i) 
    {
        if (args[i].kwarg_only() && !seen_kwarg_only) 
        {
            TORCH_CHECK(false, "Not supported yet kwarg fro declaration.")
            seen_kwarg_only = true;
        }
        if (args[i].type()->isSubtypeOf(TensorType::get()))
        {
            oss << "Tensor " << args[i].name();
        }
        else 
        {
            TORCH_CHECK(false, "Not supported argument type.");
        }
        if (i != args.size() - 1)
        {
            oss << ", ";
        }
    }
    oss << ") {\n";
    return oss.str();
}

std::string Ultra::returns()
{
    const auto& returns = schema_ -> returns();
    TORCH_CHECK(returns.size() == 1, "Currently supported number of outputs is one.");
    const auto& r = returns.at(0);
    if (r.type()->isSubtypeOf(ListType::ofTensors())) 
    {
        std::cout << "MURAD" << std::endl;
    }
    else
    {
        TORCH_CHECK(false, "Not supported return type.");
    }
    std::cout << "R: " << r << std::endl;
    return "";
}

void Ultra::writeCMakeListsTxt()
{
    auto w = fs::current_path();
    w /= "w";
    w /= "CMakeLists.txt";
    std::ofstream(w) << "cmake_minimum_required(VERSION 3.19 FATAL_ERROR)\n"
                     << "project(Synthetic CXX)\n\n"
                     << "set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../ext/libtorch)\n"
                     << "find_package(Torch REQUIRED)\n\n"
                     << "set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}\")\n"
                     << "set(CMAKE_CXX_STANDARD 17)\n"
                     << "set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -std=c++17 -O3\")\n\n"
                     << "set(SYN_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/synthetic_lib)\n\n"
                     << "set(SOURCE_FILES forward.h forward.cpp)\n"
                     << "add_library(Synthetic SHARED STATIC ${SOURCE_FILES})\n"
                     << "target_link_libraries(Synthetic \"${TORCH_LIBRARIES}\")\n"
                     << "install(TARGETS Synthetic DESTINATION ${SYN_INSTALL_LIB_DIR})\n"
                     << "install(FILES forward.h DESTINATION ${SYN_INSTALL_LIB_DIR})";
}

void Ultra::syntheticLib()
{
    auto w = fs::current_path();
    w /= "w";

    std::ostringstream cmd;
    cmd << "cd " << w.c_str() << "; cmake . ; make ; make install; cd - ";
    std::system(cmd.str().c_str());
}

} 