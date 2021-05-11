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

namespace ultra 
{

using namespace torch::jit;

std::unordered_map<std::string, std::string> Ultra::s_natives_ = 
{
    {"add", "add"},
    {"mul", "mul"},
    {"clamp", "clamp"},
    {"transpose", "transpose"},
    {"bmm", "bmm_cpu"},
    {"flatten", "flatten"},
    {"cat", "_cat_cpu"},
    {"addmm", "addmm_cpu"},
    {"sigmoid", "sigmoid"},
    {"conv2d", "conv2d"},
    {"relu", "relu"},
    {"matmul", "matmul"},
    {"max_pool2d", "max_pool2d"},
    {"batch_norm", "batch_norm"},
    {"adaptive_avg_pool2d", "adaptive_avg_pool2d_cpu"}
};

std::unordered_map<std::string, std::string> Ultra::s_native_outs_ = 
{
    {"add", "add_out"},
    {"mul", "mul_out"},
    {"clamp", "clamp_out"},
    {"bmm", "bmm_out_cpu"},
    {"cat", "_cat_out_cpu"},
    {"addmm", "addmm_cpu_out"},
    {"sigmoid", "sigmoid_out"},
    {"matmul", "matmul_out"},
    {"adaptive_avg_pool2d", "adaptive_avg_pool2d_out_cpu"}
};


int Ultra::s_phiLoop_id_ = 0;

Ultra::Ultra(script::Module module) 
                    : module_(module.copy())
                    , graph_(nullptr)
                    , schema_(nullptr)
                    , declaration_("")
                    , code_("")
                    , first_time_(true)
{
    try 
    {
        // Set eval mode
        module_.eval();
        // Freeze for optimization
        module_ = freeze_module(module_);
        // Get forward method's graph representation
        graph_ = module_.get_method("forward").graph();
        // Extract function schema
        const c10::FunctionSchema& s = module_.get_method("forward").function().getSchema();
        std::vector<Argument> args({s.arguments().begin() + 1, s.arguments().end()});
        schema_ =  std::make_unique<c10::FunctionSchema>(s.cloneWithArguments(args));
        // Prepare for code generation
        OptimizeGraph();
        RemoveSelfFromGraphInput();
        // Some foldings which helps 
        // to more easly generate code
        foldSizeLenGT();
        foldDimNE();
        foldDimEQ();
        // Dump
        graph_ -> dump();
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

void Ultra::OptimizeGraph()
{
    // Apply optimization passes from jit/passes
    Inline(*graph_);
    ConstantPropagation(graph_);
    Canonicalize(graph_);
    ConstantPropagation(graph_);
    RemoveTensorMutation(graph_);
    ConstantPropagation(graph_);
    EliminateDeadCode(graph_);
    ConstantPropagation(graph_);
}

void Ultra::RemoveSelfFromGraphInput()
{
    if (graph_ -> inputs().at(0) -> type() -> is_module()) 
    {
        TORCH_CHECK(!graph_ -> inputs().at(0) -> hasUses());
        graph_ -> eraseInput(0);
    }
}

void Ultra::writeHPP(const fs::path& w)
{
    fs::path p = w;
    p /= "forward.h";
    std::ofstream(p) << "#include <ATen/ATen.h>\n"
                     << "#include <vector>\n"
                     << "#include <tuple>\n\n"
                     << "using namespace at;\n\n"
                     << declaration_ << "\n";
}
void Ultra::writeCPP(const fs::path& w)
{
    fs::path p = w;
    p /= "forward.cpp";
    std::ofstream(p) << code_;
}

std::optional<std::string> Ultra::buildLibrary(const fs::path& w)
{
    // Generate source code
    graphTraversal();

    if (w.empty())
    {
        // For testing
        return std::make_optional<std::string>(code_);
    }
    // write to files
    writeCPP(w);
    writeHPP(w);
    return std::nullopt;
}

void Ultra::graphTraversal() 
{
    std::ostringstream fwd;
    fwd << "\n\n";
    // Get outputs
    auto outputs = graph_ -> block() -> outputs();
    // Return type
    if (outputs.size() == 1) 
    {
        fwd << type(outputs[0] -> type());
    }
    else 
    {   
        // aggregate outputs into tuple
        fwd << "std::tuple<";
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            fwd << *outputs[i] -> type();
            if (i != outputs.size() - 1)
            {
                fwd << ", ";
            }
        }
        fwd << ">";
    }

    // Generated function name
    fwd << " synthetic_forward (";
    // Get inputs
    auto inputs = graph_ -> block() -> inputs();
    // Generate arguments for synthetic_forward
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        fwd << *inputs[i] -> type() << "& " << normalizeName(inputs[i] -> debugName());
        if (i != inputs.size() - 1)
        {
            fwd << ", ";
        }
    }
    // Save declaration to write in hpp file
    declaration_ = fwd.str() + ");";
    // Continute cpp part
    fwd << ") \n{\n";
    // Add guard to optimize runtime
    // TODO for new versions use "InferenceMode guard(true)"
    fwd << "\tNoGradGuard no_grad;\n\n";

    /*
     * if (first_time) {
     *  use either at::naitve or at version
     *  first_time = false;
     * } else {
     *  try to use 'out' version
     * }
     */
    size_t level = 1;
    fwd << std::string(2 * level, ' ');
    fwd << "if (first_time) {\n";
    fwd << blockTraversal(graph_ -> block(), level + 1);
    fwd << std::string(2 * (level + 1), ' ');
    fwd << "first_time = false;\n";
    fwd << std::string(2 * level, ' ');
    fwd << "} else {\n";
    first_time_ = false;
    fwd << blockTraversal(graph_ -> block(), level + 1);
    fwd << std::string(2 * level, ' ');
    fwd << "}\n";
    fwd << std::string(2 * level, ' ');
    // Generate last 'return' statement
    if (outputs.size() == 1) 
    {
        if (outputs[0] -> type() -> isSubtypeOf(ListType::ofTensors()) or
            outputs[0] -> type() -> kind() == TupleType::Kind or 
            outputs[0] -> type() -> kind() == TensorType::Kind)
        {
            fwd << "return " << normalizeName(outputs[0] -> debugName()) << ";\n";
        }
        else 
        {
            TORCH_CHECK(false, "Not supported return type.");
        }
    }
    else 
    {
        fwd << "return std::make_tuple(";
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            fwd << normalizeName(outputs[i] -> debugName());
            if (i != outputs.size() - 1)
            {
                fwd << ", ";
            }
        }
        fwd << ");\n";
    }
    fwd << "}\n";
    // Includes
    std::ostringstream includes;
    includes << "#include \"forward.h\"\n";
    includes << "#include <ATen/core/grad_mode.h>\n\n";
    // Declare global flag for caching
    std::ostringstream gScope;
    gScope << "bool first_time = true;\n\n";
    for (const auto & e : global_scope_)
    {
        gScope << e << "\n";
    }
    // Aggregate all parts together 
    code_ = includes.str() + gScope.str() + fwd.str();
}
 
// Make names c++ friendly
std::string Ultra::normalizeName(const std::string& name) const
{
    auto s = 'g' + name;
    for (auto & c : s)
    {
        if (c == '.')
        {
            c = '_';
        }
    }
    return s;
}

std::string Ultra::type(const at::TypePtr& tPtr)
{
    if (tPtr -> isSubtypeOf(ListType::ofTensors())) 
    {
        return "std::vector<Tensor> ";
    } 
    else if (tPtr -> kind() == TupleType::Kind)
    {
        std::ostringstream oss;
        oss << "std::tuple<";
        for (size_t i = 0; i < tPtr -> containedTypes().size(); ++i)
        {
            if (tPtr -> containedTypes().at(i) -> kind() == TensorType::Kind)
            {
                oss << "Tensor";
            }
            else 
            {
                std::stringstream msg;
                msg << "Not supported tuple's element type " << *(tPtr -> containedTypes().at(i)) << "\n";
                AT_ERROR(msg.str());
            }
            if (i != tPtr -> containedTypes().size() - 1) 
            {
                oss << ", ";
            }
        }
        oss << "> ";
        return oss.str();
    }
    else if (tPtr -> isSubtypeOf(TensorType::get()))
    {
        return "Tensor";
    }
    std::stringstream msg;
    msg << "Add support for " << (*tPtr) << "\n";
    AT_ERROR(false, msg.str());
    return "";
}

std::string Ultra::primNode(Node* node, size_t level)
{
    std::ostringstream oss;
    
    if (node -> kind() == prim::Constant)
    {
        if (node -> hasAttributes()) 
        {
            // Add constants into global scope which will help to
            // declare/define constants only once and use for all 
            // 'forward' calls
            // Example: 
            // Input:  %6 : float = prim::Constant[value=10.]()
            // Output: const float g6 = 10;
            global_scope_.insert(primConstantAttributes(node, 0));
        } 
        else
        {
            // None = prim::Constant()
            // this cases handled in handleNonePrimConstant
        }
        return "";
    }
    else if (node -> kind() == prim::ListConstruct or node -> kind() == prim::TupleConstruct)
    {
        // Converts 
        // prim::ListConstruct into std::vector 
        // prim::TupleConstruct into std::tuple
        oss << std::string(2 * level, ' ');
        TORCH_CHECK(node -> outputs().size() == 1, 
            "Expected number of outputs for prim::ListConstruct is one.");
        auto out = node -> outputs()[0];

        std::stringstream ss;
        ss << type(out -> type()) << normalizeName(out -> debugName()) << ";\n";
        global_scope_.insert(ss.str());

        oss << normalizeName(out -> debugName()) << " = {";
        for (size_t i = 0; i < node -> inputs().size(); ++i) 
        {
            oss << normalizeName(node -> inputs()[i] -> debugName());
            if (i != node -> inputs().size() - 1) 
            {
                oss << ", ";
            }
        }
        oss << "};\n";
        return oss.str();
    }
    else if (node -> kind() == prim::RaiseException)
    {
        oss << std::string(2 * level, ' ');
        oss << "throw;\n";
        return oss.str();
    }
    else if (node -> kind() == prim::PythonOp) 
    {
        TORCH_CHECK(false, "prim::PythonOp is not supported yet");
    } 
    else if (node -> hasAttribute(attr::Subgraph))
    {
        TORCH_CHECK(false, "attr::Subgraph is not supported yet");
    } 
    else if (std::string(node -> kind() . toQualString()) == "Ultra::size_len_gt")
    {
        // Converts %a : bool = Ultra::size_len_gt(%b, %c) into bool a = b.ndimension() > c;
        // TODO probably there is aten::native version for ndimension()
        oss << std::string(2 * level, ' ');
        auto out = node -> outputs()[0];

        oss << "bool " << normalizeName(out -> debugName()) << " = "
            << normalizeName(node -> inputs()[0] -> debugName()) << ".ndimension() > " 
            << normalizeName(node -> inputs()[1] -> debugName()) << ";\n";
        return oss.str();
    }
    else if (std::string(node -> kind() . toQualString()) == "Ultra::dim_ne")
    {
        // Converts %a : bool = Ultra::dim_ne(%b, %c) into bool a = b.ndimension() != c;
        // TODO probably there is aten::native version for ndimension()
        oss << std::string(2 * level, ' ');
        auto out = node -> outputs()[0];

        oss << "bool " << normalizeName(out -> debugName()) << " = "
            << normalizeName(node -> inputs()[0] -> debugName()) << ".ndimension() != " 
            << normalizeName(node -> inputs()[1] -> debugName()) << ";\n";
        return oss.str();
    }
    else if (std::string(node -> kind() . toQualString()) == "Ultra::dim_eq")
    {
        // Converts %a : bool = Ultra::dim_eq(%b, %c) into bool a = b.ndimension() == c;
        // TODO probably there is aten::native version for ndimension()
        oss << std::string(2 * level, ' ');
        auto out = node -> outputs()[0];

        oss << "bool " << normalizeName(out -> debugName()) << " = "
            << normalizeName(node -> inputs()[0] -> debugName()) << ".ndimension() == " 
            << normalizeName(node -> inputs()[1] -> debugName()) << ";\n";
        return oss.str();
    }

    if (node -> hasAttributes()) 
    {
        TORCH_CHECK(false, "node -> hasAttributes()is not supported yet");
    }

    // Generate primitive instruction 
    if (allArgsScalar(node))
    {
        // TODO not implemented yet
        // if op's arguments are scalar no need to use aten library
        // oss << scalarArgsOp(node, level);
    }
    if (first_time_) 
    {
        // For the first call try to use aten::native verion
        // which will reduce PyTorch dispatcher overhead
        oss << atNative(node, level);
    }
    else 
    {
        // If there is equivalent 'out' version use if, 
        // it will help to reduce allocation/deallocation
        // of output tensors which have created during first call
        oss << atNativeOut(node, level);
    }

    return oss.str();
}

bool Ultra::allArgsScalar(Node* node)
{
    // TODO not implemented yet
    return false;
    // Checks if all arguments are integral types
    auto primIns = node -> inputs();
    size_t inSize = primIns.size();
    for (size_t i = 0; i < inSize; ++i) 
    {
        // try to fetch type from argument's attribute
        // std::cerr << primIns[i] -> node() -> hasAttributes() << std::endl;
        bool b1 = primIns[i] -> type() -> kind() == TypeKind::BoolType;
        bool b2 = primIns[i] -> type() -> kind() == TypeKind::FloatType;
        bool b3 = primIns[i] -> type() -> kind() == TypeKind::IntType;
        if (not b1 or not b2 or not b3)
        {
            return false;
        }
    }
    return true;
}

std::string Ultra::atNative(Node* node, size_t level)
{
    // Here aten::native functions will be generated 
    // without considering out variants , 
    // since this is first time call.
    // Output of this OPs is a newly created tensors
    // and will be assigned into globally visible tensor variables
    // Which allow us to reuse in subsequent calls.
    // Tensor x;
    // if (first_time)
    // {
    //     x = aten::add(...);
    // } 
    // else
    // {
    //     aten::add_out(x, ...);
    // }
    std::ostringstream oss;

    auto outs = node -> outputs();
    std::stringstream msg;
    msg << node -> kind() . toQualString() << " has " << outs.size() << " expected 1";
    TORCH_CHECK(outs.size() == 1, msg.str());
    
    auto primOut = outs[0];
    
    // Gets declaration of OP's output
    // For example if %a = aten::add(....) then
    // in the global scope we will generate: "Tensor g_a;"
    std::stringstream ss;
    ss << *primOut -> type() << " " << normalizeName(primOut -> debugName()) << ";\n";
    global_scope_.insert(ss.str());

    // Gets OP's C++ equivalent API
    oss << std::string(2 * level, ' ');
    oss << normalizeName(primOut -> debugName()) << " = ";
    if (s_natives_.count(std::string(node -> kind() . toUnqualString())))
    {
        oss << "native::" << s_natives_[std::string(node -> kind() . toUnqualString())] << " (";
    } 
    else 
    {
        oss << node -> kind() . toUnqualString() << " (";
    }
    
    // Process inputs.
    // Since input is uniquely identified , 
    // and graph is SSAed, task is here 
    // just line them up.
    auto primIns = node -> inputs();
    size_t inSize = primIns.size();

    for (size_t i = 0; i < inSize; ++i)
    {
        // Case where input is None
        handleNonePrimConstant(node, i);
        oss << normalizeName(primIns[i] -> debugName());
        if (i != inSize - 1)
        {
            oss << ", ";
        }
    }
    oss << ");\n";
    return oss.str();
}

std::string Ultra::atNativeOut(Node* node, size_t level)
{
    // Try to use 'out' variant of OP
    // in that case we will reduce number 
    // of allocs/deallocs for tensors.
    std::ostringstream oss;

    auto outs = node -> outputs();
    std::stringstream msg;
    msg << node -> kind() . toQualString() << " has " << outs.size() << " expected 1";
    TORCH_CHECK(outs.size() == 1, msg.str());

    auto primOut = outs[0];
    auto primIns = node -> inputs();
    size_t inSize = primIns.size();
    
    // Prefer native out version of op
    if (s_native_outs_.count(std::string(node -> kind() . toUnqualString())) && extraConditionsOn(node))
    {
        oss << std::string(2 * level, ' ');
        oss << "native::" << s_native_outs_[std::string(node -> kind() . toUnqualString())] << " (";
        oss << normalizeName(primOut -> debugName());
        if (inSize > 0) {
            oss << ", ";
        }
    }
    // Prefer native version
    else if (s_natives_.count(std::string(node -> kind() . toUnqualString())))
    {
        oss << std::string(2 * level, ' ');
        oss << normalizeName(primOut -> debugName()) << " = ";
        oss << "native::" << s_natives_[std::string(node -> kind() . toUnqualString())] << " (";
    }
    // If we failed to get 'native out' and 'native' versions go with sandard one
    else 
    {
        oss << std::string(2 * level, ' ');
        oss << normalizeName(primOut -> debugName()) << " = ";
        oss << node -> kind() . toUnqualString() << " (";
    }
    
    // Process inputs.
    // Since input is uniquely identified , 
    // and graph is SSAed, task is here 
    // just line them up.
    for (size_t i = 0; i < inSize; ++i)
    {
        // Case where input is None
        handleNonePrimConstant(node, i);
        oss << normalizeName(primIns[i] -> debugName());
        if (i != inSize - 1)
        {   
            oss << ", ";
        }
    }
    oss << ");\n";
    return oss.str();
}

bool Ultra::extraConditionsOn(Node* node)
{
    // second input should be tensor for 'mul' node
    if (std::string(node -> kind() . toUnqualString()) == "mul") 
    {
        return node->inputs()[1]->type()->kind() == TypeKind::TensorType ? true : false;
    }
    return true;
}

void Ultra::handleNonePrimConstant(Node* node, size_t arg_index) 
{
    if (node -> inputs()[arg_index] -> node() -> kind() != prim::Constant)
    {
        return ;
    }
    if (node -> inputs()[arg_index] -> node() -> hasAttributes()) 
    {
        return ;
    }
    // We have  %x : None = prim::Constant() 

    // Get underlying type from FunctionSchema and map to c++ equivalent 
    const FunctionSchema* the_schema = node -> maybeSchema();
    if (the_schema) {
        std::ostringstream oss;
        const Argument arg = the_schema -> arguments()[arg_index];
        if (arg.type() -> kind() == TypeKind::OptionalType)
        {
            if (arg.type() -> cast<OptionalType>() -> getElementType() -> kind() == TypeKind::TensorType)
            {
                oss << "const Tensor " 
                    << normalizeName(node -> inputs()[arg_index] -> node() -> outputs()[0] -> debugName()) 
                    << " = {};\n";
                global_scope_.insert(oss.str());
            }
            else 
            {
                oss << "c10::optional<ScalarType> " 
                    << normalizeName(node -> inputs()[arg_index] -> node() -> outputs()[0] -> debugName()) 
                    << " = c10::nullopt;\n";
                global_scope_.insert(oss.str());
            }
        } 
        else
        {
            node -> dump();
            TORCH_CHECK(false, "NonePrimConstant not supported yet.");
        }
    } 
    else
    {
        node -> dump();
        std::stringstream ss;
        ss << node -> kind() . toQualString() << " has no schema.";
        TORCH_CHECK(false, ss.str());
    }

}

std::string Ultra::phiNode(Node* node, size_t level)
{
    std::ostringstream oss;

    if (node -> kind() == prim::Loop) 
    {
        // Converts prim::Loop into while loop statement
        int id = ++s_phiLoop_id_; // can be more than one loops in owner block
        TORCH_CHECK(node -> blocks().size() == 1, "expected 1 block for prim::Loop");
        auto primLoopBlock = node -> blocks()[0];
        auto blockInputs = node -> blocks()[0] -> inputs();
        auto blockOutputs = node -> blocks()[0] -> outputs();
        auto trip_count = normalizeName(blockInputs[0] -> debugName());

        auto max_trip_count = node -> input(0);
        auto initial_condition = node -> input(1);
        for (size_t i = 2; i < node -> inputs().size(); ++i)
        {
            oss << std::string(2 * level, ' ');
            oss << *(node -> input(i)) -> type() << " " << normalizeName(node -> output(i - 2) -> debugName())
                << " = " << normalizeName(node -> input(i) -> debugName()) << ";\n";
        }
        oss << std::string(2 * level, ' ');
        oss << *initial_condition -> type() << " prim_loop_condition_" << id 
            << " = " << normalizeName(initial_condition -> debugName()) << ";\n";
        oss << std::string(2 * level, ' ');

        oss << *max_trip_count -> type() << " " << trip_count << " = 0;\n";
        oss << std::string(2 * level, ' ');
        oss << "while (prim_loop_condition_" << id 
            << " and " << trip_count << " < " << normalizeName(max_trip_count -> debugName())
            << ") {\n";
        ++level;
        for (size_t i = 1; i < blockInputs.size(); ++i)
        {
            oss << std::string(2 * level, ' ');
            oss << normalizeName(blockInputs[i] -> debugName()) << " = " << normalizeName(node -> output(i - 1) -> debugName()) << ";\n";
        }
        
        oss << blockTraversal(primLoopBlock, level);

        for (size_t i = 1; i < blockOutputs.size(); ++i)
        {
            oss << std::string(2 * level, ' ');
            oss << normalizeName(node -> output(i - 1) -> debugName()) << " = " << normalizeName(blockOutputs[i] -> debugName()) << ";\n";
        }
        oss << std::string(2 * level, ' ');
        oss << "prim_loop_conditions_" << id << " = " << normalizeName(blockOutputs[0] -> debugName()) << ";\n";
        --level;
        oss << std::string(2 * level, ' ');
        oss << "}\n";
    } 
    else 
    {
        TORCH_CHECK(node -> kind() == prim::If, "Expected phiNodes are prim::Loop and prim::if, but got " + std::string(node -> kind() . toQualString()));
        auto outputs = node -> outputs();
        auto condition = node -> inputs()[0];
        TORCH_CHECK(node -> inputs().size() == 1, "Expected one input for prim::If");
        if (node -> blocks().size() == 2)  // if - else 
        {
            // Declare block outputs
            for (size_t i = 0; i < outputs.size(); ++i)
            {
                std::stringstream ss;
                ss << type(outputs[i] -> type()) << " " << normalizeName(outputs[i] -> debugName()) << ";\n";
                global_scope_.insert(ss.str());
            }

            auto trueBlock = node -> blocks()[0];
            auto falseBlock = node -> blocks()[1];
            oss << std::string(2 * level, ' ');
            oss << "if (" << normalizeName(condition -> debugName()) << ") {\n";
            ++level;
            // True branch

            oss << blockTraversal(trueBlock, level);

            for (size_t i = 0; i < outputs.size(); ++i)
            {
                oss << std::string(2 * level, ' ');
                oss << normalizeName(outputs[i] -> debugName()) << " = " << normalizeName(trueBlock -> outputs()[i] -> debugName()) << ";\n";
            }
            --level;
            oss << std::string(2 * level, ' ');
            oss << "} else {\n";
            ++level;
            // False branch
            
            oss << blockTraversal(falseBlock, level);

            for (size_t i = 0; i < outputs.size(); ++i)
            {
                oss << std::string(2 * level, ' ');
                oss << normalizeName(outputs[i] -> debugName()) << " = " << normalizeName(falseBlock -> outputs()[i] -> debugName()) << ";\n";
            }
            --level;
            oss << std::string(2 * level, ' ');
            oss << "}\n";
        }
        else if (node -> blocks().size() == 1) // if
        {
            // Pass (converted into if-else)
        } 
        else 
        {
            TORCH_CHECK(false, "Not supported yet");
        }
    }
    return oss.str();
}

std::string Ultra::blockTraversal(Block* block, size_t level)
{
    std::ostringstream oss;

    // Process block's nodes
    for (auto node : block -> nodes())
    {
        if (0 == node -> blocks().size())
        {
            // Process primitive node (simple instruction)
            oss << primNode(node, level);
        }
        else
        {
            // Process control flow instructions
            oss << phiNode(node, level);
        }
    }
    return oss.str();
}

std::string Ultra::ivalue(const Value* output, const IValue& v, size_t level)
{
    std::ostringstream oss;

    if (v.isIntList())
    {
        auto list = v.toList();

        oss << std::string(2 * level, ' ');
        oss << "const std::array<int64_t, " << list.size() << "> " << normalizeName(output -> debugName()) << " = {";
        for (size_t i = 0; i < list.size(); ++i)
        {
            oss << list[i];
            if (i != list.size() - 1)
            {
                oss << ", ";
            }
        }
        oss << "};\n";
        return oss.str();
    } else  {
        AT_ERROR("Tag not supported yet: ", v.tagKind());
    }
    return "";
}

std::tuple<std::string, std::string> Ultra::cppTypeFrom(const at::ScalarType& st)
{
    switch (st) 
    {
        case at::ScalarType::Byte:
        {
            return {"uint8_t", "kByte"};
        }
        case at::ScalarType::Char:
        {
            return {"int8_t", "kChar"};
        }
        case at::ScalarType::Short:
        {
            return {"int16_t", "kShort"};
        }
        case at::ScalarType::Int:
        {
            return {"int", "kInt"};
        }
        case at::ScalarType::Long:
        {
            return {"int64_t", "kLong"};
        }
        case at::ScalarType::Half:
        {
            return {"at::Half", "kHalf"};
        }
        case at::ScalarType::Float:
        {
            return {"float", "kFloat"};
        }
        case at::ScalarType::Double:
        {
            return {"double", "kDouble"};
        }
        case at::ScalarType::ComplexHalf:
        {
            return {"c10::complex<c10::Half>", "TODO"};
        }
        case at::ScalarType::ComplexFloat:
        {
            return {"c10::complex<float>", "TODO"};
        }
        case at::ScalarType::ComplexDouble:
        {
            return {"c10::complex<double>", "TODO"};
        }
        case at::ScalarType::Bool:
        {
            return {"bool", "kBool"};
        }
        case at::ScalarType::QInt8:
        {
            return {"c10::qint8", "TODO"};
        }
        case at::ScalarType::QUInt8:
        {
            return {"c10::quint8", "TODO"};
        }
        case at::ScalarType::QInt32:
        {
            return {"c10::qint32", "TODO"};
        }
        case at::ScalarType::BFloat16:
        {
            return {"at::BFloat16", "TODO"};
        }
        case at::ScalarType::NumOptions:
        {
            return {"NumOptions", "TODO"};
        }
        case at::ScalarType::Undefined:
        {
            return {"Undefined", "Undefined"};
        }
    }
    TORCH_CHECK(false, "Unknown scalar type.");
    return {"Unknown", "Unknown"};
}

std::string Ultra::constantInitData(const at::Tensor& tensor)
{
    std::ostringstream oss;
     oss << "({";

    switch (tensor.scalar_type()) 
    {
        case at::ScalarType::Byte:
        {
            auto d = tensor.data_ptr<uint8_t>();
            for (size_t i = 0; i < tensor.numel(); ++i)
            {
                oss << d[i];
                if (i != tensor.numel() - 1)
                {
                    oss << ", ";
                }
            }
            break;
        }
        case at::ScalarType::Char:
        {
            auto d = tensor.data_ptr<int8_t>();
            for (size_t i = 0; i < tensor.numel(); ++i)
            {
                oss << d[i];
                if (i != tensor.numel() - 1)
                {
                    oss << ", ";
                }
            }
            break;
        }
        case at::ScalarType::Short:
        {
            auto d = tensor.data_ptr<int16_t>();
            for (size_t i = 0; i < tensor.numel(); ++i)
            {
                oss << d[i];
                if (i != tensor.numel() - 1)
                {
                    oss << ", ";
                }
            }
            break;
        }
        case at::ScalarType::Int:
        {
            auto d = tensor.data_ptr<int>();
            for (size_t i = 0; i < tensor.numel(); ++i)
            {
                oss << d[i];
                if (i != tensor.numel() - 1)
                {
                    oss << ", ";
                }
            }
            break;
        }
        case at::ScalarType::Long:
        {
            auto d = tensor.data_ptr<int64_t>();
            for (size_t i = 0; i < tensor.numel(); ++i)
            {
                oss << d[i];
                if (i != tensor.numel() - 1)
                {
                    oss << ", ";
                }
            }
            break;
        }
        case at::ScalarType::Half:
        {
            auto d = tensor.data_ptr<at::Half>();
            for (size_t i = 0; i < tensor.numel(); ++i)
            {
                oss << d[i];
                if (i != tensor.numel() - 1)
                {
                    oss << ", ";
                }
            }
            break;
        }
        case at::ScalarType::Float:
        {
            auto d = tensor.data_ptr<float>();
            for (size_t i = 0; i < tensor.numel(); ++i)
            {
                oss << d[i];
                if (i != tensor.numel() - 1)
                {
                    oss << ", ";
                }
            }
            break;
        }
        case at::ScalarType::Double:
        {
            auto d = tensor.data_ptr<double>();
            for (size_t i = 0; i < tensor.numel(); ++i)
            {
                oss << d[i];
                if (i != tensor.numel() - 1)
                {
                    oss << ", ";
                }
            }
            break;
        }
        case at::ScalarType::ComplexHalf:
        {
            TORCH_CHECK(false, "c10::complex<c10::Half> not supported");
            return "";
        }
        case at::ScalarType::ComplexFloat:
        {
            auto d = tensor.data_ptr<c10::complex<float>>();
            for (size_t i = 0; i < tensor.numel(); ++i)
            {
                oss << d[i];
                if (i != tensor.numel() - 1)
                {
                    oss << ", ";
                }
            }
            break;
        }
        case at::ScalarType::ComplexDouble:
        {
            auto d = tensor.data_ptr<c10::complex<double>>();
            for (size_t i = 0; i < tensor.numel(); ++i)
            {
                oss << d[i];
                if (i != tensor.numel() - 1)
                {
                    oss << ", ";
                }
            }
            break;
        }
        case at::ScalarType::Bool:
        {
            auto d = tensor.data_ptr<bool>();
            for (size_t i = 0; i < tensor.numel(); ++i)
            {
                oss << d[i];
                if (i != tensor.numel() - 1)
                {
                    oss << ", ";
                }
            }
            break;
        }
        case at::ScalarType::QInt8:
        {
            TORCH_CHECK(false, "c10::qint8 not supported");
            return "";
        }
        case at::ScalarType::QUInt8:
        {
            TORCH_CHECK(false, "c10::quint8 not supported");
            return "";
        }
        case at::ScalarType::QInt32:
        {
            TORCH_CHECK(false, "c10::qint32 not supported");
            return "";
        }
        case at::ScalarType::BFloat16:
        {
            auto d = tensor.data_ptr<at::BFloat16>();
            for (size_t i = 0; i < tensor.numel(); ++i)
            {
                oss << d[i];
                if (i != tensor.numel() - 1)
                {
                    oss << ", ";
                }
            }
            break;
        }
        case at::ScalarType::NumOptions:
        {
            TORCH_CHECK(false, "Not implemented NumOptions for Tensor init.");
            return "NumOptions";
        }
        case at::ScalarType::Undefined:
        {
            TORCH_CHECK(false, "Undefined");
            return "Undefined";
        }
    }

    oss << "})";
    return oss.str();
}

std::string Ultra::primConstantTensor(const at::Tensor& t, const Node* node, size_t level) 
{
    std::ostringstream oss;

    if (not t.defined()) 
    {
        oss << std::string(2 * level, ' ');
        oss << "const Tensor " << normalizeName(node -> outputs()[0] -> debugName()) << ";\n";
        return oss.str();
    }
    else if (t.is_sparse())
    {
        TORCH_CHECK(false, "Sparse Tensor is not supported yet.");
        return "";
    }

    at::Tensor tensor = t;
    if (t.is_quantized())
    {
        tensor = t.dequantize().to(at::kCPU, at::kDouble).contiguous();
    }
    else if (t.is_mkldnn())
    {
        tensor = t.to_dense().to(at::kCPU, at::kDouble).contiguous();
    }
    else 
    {
        tensor = t.to(at::kCPU, at::kDouble).contiguous();
    }
    if (tensor.ndimension() == 0) 
    {
        TORCH_CHECK(false, "Tensor dim 0 is not supported yet.");
    }
    else 
    {
        std::string initList = "";
        if (tensor.numel() > 0) 
        {
            initList = constantInitData(tensor);
        }
        if (t.scalar_type() == at::ScalarType::Undefined)
        {
            TORCH_CHECK(false, "Undefined scalar type.");
        }
        std::string cppType, atType;
        std::tie(cppType, atType) = cppTypeFrom(t.scalar_type());

        oss << std::string(2 * level, ' ');
        oss << "std::vector<" << cppType << "> " << normalizeName(node -> outputs()[0] -> debugName()) << "_v " << initList << ";\n";
        oss << std::string(2 * level, ' ');
        oss << "auto " << normalizeName(node -> outputs()[0] -> debugName()) << "_opts = TensorOptions().dtype("<< atType << ");\n";
        oss << std::string(2 * level, ' ');
        oss << "const Tensor " << normalizeName(node -> outputs()[0] -> debugName()) << " = from_blob(" 
            << normalizeName(node -> outputs()[0] -> debugName()) << "_v.data(), {";
        size_t s = tensor.sizes().size();
        for (size_t i = 0; i < s; ++i)
        {
            oss << tensor.sizes()[i];
            if (i != s - 1)
            {
                oss << ", ";
            }
        }
        oss << "}, " << normalizeName(node -> outputs()[0] -> debugName()) << "_opts);\n";
    }
    if (t.is_quantized())
    {
        TORCH_CHECK(false, "Quantized Tensor is not supported yet.");
    }
    return oss.str();
}

std::string Ultra::primConstantAttributesValue(const Node* node, const Symbol& name, size_t level)
{
    std::ostringstream oss;

    auto outputs = node -> outputs();
    TORCH_CHECK(outputs.size() == 1, "Expected size of outputs of primConstant is 1");
    auto output = outputs[0];

    switch (node -> kindOf(name)) 
    {
        case AttributeKind::f:
        {
            const double& f = node -> f(name);
            oss << std::string(2 * level, ' ');
            oss << "const " << *(output -> type()) << " " << normalizeName(output -> debugName()) << " = " << f << ";\n";
            break;
        }
        case AttributeKind::fs:
        {
            const std::vector<double>& fs = node -> fs(name);
            oss << std::string(2 * level, ' ');
            oss << "const " << *(output -> type()) << " " << normalizeName(output -> debugName()) << "["  << fs.size() << "] = {";
            for (size_t i = 0; i < fs.size(); ++i)
            {
                oss << fs[i];
                if (i != fs.size() - 1)
                {
                    oss << ", ";
                }
            }
            oss << "};\n";
            break;
        }
        case AttributeKind::i:
        {
            const int64_t i = node -> i(name);
            oss << std::string(2 * level, ' ');
            oss << "const " << *(output -> type()) << " " << normalizeName(output -> debugName()) << " = " << i << ";\n";
            break;
        }
        case AttributeKind::is:
        {
            const std::vector<int64_t> is = node -> is(name);
            oss << std::string(2 * level, ' ');
            oss << "const " << *(output -> type()) << " " << normalizeName(output -> debugName()) << "["  << is.size() << "] = {";
            for (size_t i = 0; i < is.size(); ++i)
            {
                oss << is[i];
                if (i != is.size() - 1)
                {
                    oss << ", ";
                }
            }
            oss << "};\n";
            break;
        }
        case AttributeKind::s:
        {
            const std::string s = node -> s(name);
            oss << std::string(2 * level, ' ');
            oss << "const std::string " << normalizeName(output -> debugName()) << " = \"" << s << "\";\n";
            break;
        }
        case AttributeKind::ss:
        {
            const std::vector<std::string> ss = node -> ss(name);
            TORCH_CHECK(false, "Attribute vector<string> value is not supported.");
            break;
        }
        case AttributeKind::t:
        {
            const at::Tensor& t = node -> t(name);
            oss << primConstantTensor(t, node, level);
            break;
        }
        case AttributeKind::ts:
        {
            TORCH_CHECK(false, "Attribute [<Tensor>] value is not supported.")
            break;
        }
        case AttributeKind::ival:
        {
            const c10::IValue& ival = node -> ival(name);
            return ivalue(output, ival, level);
        }
        case AttributeKind::g:
        {
            TORCH_CHECK(false, "Attribute <Graph> value is not supported.")
            break;
        }
        case AttributeKind::gs:
        {
            TORCH_CHECK(false, "Attribute [<Graphs>] value is not supported.")
            break;
        }
        case AttributeKind::ty:
        {
            const c10::TypePtr ty = node -> ty(name);
            TORCH_CHECK(false, "Attribute c10::TypePtr value is not supported.")
            break;
        }
        case AttributeKind::tys:
        {
            const std::vector<c10::TypePtr> tys = node -> tys(name);
            TORCH_CHECK(false, "Attribute vector<c10::TypePtr> value is not supported.")
            break;
        }
        default : 
        {
            std::cerr << "Unknown attr value" << std::endl;
            break;
        }
    }
    return oss.str();
}

std::string Ultra::primConstantAttributes(const Node* node, size_t level)
{
    std::ostringstream oss;

    auto names = node -> attributeNames();
    for (auto name : names) 
    {
        oss << primConstantAttributesValue(node, name, level);
    }
    return oss.str();
}

int forward(const std::string& workspace)
{
    try
    {
        fs::path module_path;
        for(auto& p: fs::directory_iterator(workspace))
        {
            if (p.path().extension() == fs::path(".pt"))
            {
                TORCH_CHECK(module_path.empty(), "Found more then one loadable files in " + workspace);
                module_path = p.path();
            }
        }

        TORCH_CHECK(not module_path.empty(), "Failed to find loadable module in " + workspace);
        // Deserialize the ScriptModule from a file
        script::Module module = load(module_path.c_str());
        Ultra u(module);
        u.buildLibrary(workspace);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "c10::Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

} // namespace ultra
