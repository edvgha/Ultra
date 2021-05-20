#pragma once

#include <torch/script.h>
#include <unordered_map>
#include <filesystem>

namespace ultra 
{

namespace fs = std::filesystem;

/**
 * @class Ultra
 * 
 * The PyTorch JIT IR graph compiler.
 *
 * The class will compile dynamically specified data flow PyTorch JIT IR graph into 
 * C++ which allow to reduce DL framework overhead.
 * Useful links:
 * https://github.com/pytorch/pytorch/wiki/PyTorch-IR
 * https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md
 */
class Ultra final
{
    public: // Standard methods
        /**
         * @brief Constructor
         * @param module - JIT scripted PyTorch model.
         * Load model's graph representation and prepare 
         * for c++ code generation.
         */
        explicit Ultra(torch::jit::script::Module module);
        ~Ultra() = default;
        Ultra(const Ultra&) = delete;
        Ultra(Ultra&&) = delete;
        Ultra& operator=(const Ultra&) = delete;
        Ultra& operator=(Ultra&&) = delete;

    public:
        /**
         * @brief Builds and save into the files c++ code
         * @param w - directory where generated files will be saved
         * The main steps are:
         *      1. Traverse and collect all sufficient information.
         *      2. Generate code from collected data.
         *      3. Write into the files (gnerated file names are forward.cpp/.h)
         * @note Generated forward.h will contain declartion of 'synthetic_forward'
         *       function and definition correspondingly will be in the forward.cpp file.
         *       The 'synthetic_forward' function will produce exactly the same outputs
         *       as it does PyTorch model interpreter.
         *       As final step CMake can build static library from those files, 
         *       which later can be linked and used or directly build executable
         *       (for demos this approach is used).
         */
        std::optional<std::string> buildLibrary(const fs::path& w);

    private:
        // Prepare graph
        void OptimizeGraph();
        // Self explanatory
        void RemoveSelfFromGraphInput();
        /**
         * Input:
         *          %b : int[] = aten::size(%x)
         *          %c : int = aten::len(%b)
         *          %d : bool = aten::gt(%c, %y)
         * Output:
         *          %d : bool = Ultra::size_len_gt(%x, %y) 
         */
        void foldSizeLenGT();
        /**
         * Input:
         *          %a : int = aten::dim(%x)
         *          %d : bool = aten::ne(%a, %y)
         * Output:
         *          %d : bool = Ultra::dim_ne(%x, %y)
         */
        void foldDimNE();
        /**
         * Input:
         *          %a : int = aten::dim(%x)
         *          %d : bool = aten::eq(%a, %y)
         * Output:
         *          %d : bool = Ultra::dim_eq(%x, %y)
         */
        void foldDimEQ();
        // Writes forward declaration into file
        void writeHPP(const fs::path& w);
        // Writes forward defintion and global variables into file
        void writeCPP(const fs::path& w);
        // Traverse the graph and generates code
        void graphTraversal();
        // Make names c++ friendly
        std::string normalizeName(const std::string& name) const;
        // Remove namespace if exists
        std::string removeNamespace(const std::string& name) const;
        // Process block's content
        // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#block
        std::string blockTraversal(torch::jit::Block* block, size_t level);
        // Process pimitive node (nodes which are not 'prim::If', 'prim::Loop', 'prim::With')
        // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#node
        std::string primNode(torch::jit::Node* node, size_t level);
        // Process prim::If and prim::Loop nodes
        // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#block
        // 'prim::With' is not supported yet
        std::string phiNode(torch::jit::Node* node, size_t level);
        // Try to generates aten::native function
        // otherwise standard c++ function provided by torchlib
        // for a given graph node.
        std::string atNative(torch::jit::Node* node, size_t level);
        // At first trying to generate 'aten::native functions with out version
        // if there is some, else will try to generate 'aten::native' one
        // otherwise standard c++ function provided by torchlib
        // for a given graph node.
        std::string atNativeOut(torch::jit::Node* node, size_t level);
        // Some corner case conditions for OP's overloading
        bool extraConditionsOn(torch::jit::Node* node);
        // Handles None = prim::Constant() case
        // Possible outputs are : undefined tensor or c10::nullopt
        void handleNonePrimConstant(torch::jit::Node* node, size_t arg_index);
        // True if all arugments are integral types otherwise False
        bool allArgsScalar(torch::jit::Node* node);
        // Handle prim::Constant
        // Type and constant values will be extracted and maped into c++ equivalent
        std::string primConstantAttributes(const torch::jit::Node* node, size_t level);
        // From prim::Constant Node attribute extracts type and constant value(s)
        // node - prim::Constant Node
        // name - attribute name
        std::string primConstantAttributesValue(const torch::jit::Node* node, const torch::jit::Symbol& name, size_t level);
        // Handles prim::Constant attribte when value type is Tensor
        std::string primConstantTensor(const at::Tensor& t, const torch::jit::Node* node, size_t level);
        // Handles the case when attribute is a IValue
        std::string ivalue(const torch::jit::Value* output, const c10::IValue& v, size_t level);
        // Maps TypePtr into c++ equivalent, currently only used for handling return type
        std::string type(const at::TypePtr& tPtr);
        // Returns tuple of c++ type and corresponding Torch ScalarType's string representation
        std::tuple<std::string, std::string> cppTypeFrom(const at::ScalarType& st);
        // Generates sequence of constant value(s) for initializing constants
        std::string constantInitData(const at::Tensor& tensor);
        // Customize aten::ne
        void ne();
        // Customize aten::gt
        void gt();
        // Customize aten::lt
        void lt();
        // Customize aten::dim
        void dimToSizes();
        // Map preregistered Ultra OP into C++ equivalent
        std::string mapUltraOp(const char* op);
        // Try to replace relu with relu_
        void try_to_use_inplace_relu();
        // Experimental
        std::string nodeSchema(torch::jit::Node*, size_t level);
        std::string nodeArgument(const c10::Argument& argument);
        std::string printQuotedString(const std::string& str);
        bool isPrint(char s);
        
    private:
        torch::jit::script::Module module_; // The model
        std::shared_ptr<torch::jit::Graph> graph_; // graph extracted from model
        std::unique_ptr<c10::FunctionSchema> schema_; // forward schema
        std::string declaration_;
        std::string code_; // forward.cpp file's content
        bool first_time_; // used for caching
        std::unordered_set<std::string> global_scope_; // all global declarations/definitions
        std::unordered_set<torch::jit::Node*> inplace_nodes_; // nodes should be replaced with inplace version
        // Static members
        static int s_phiLoop_id_; // used for generate unique initial loop condition variable name
};

int forward(const std::string& workspace);

} // ultra
