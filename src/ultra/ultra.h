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
 * The class will generate CPP code from given PyTorch scripted model
 */
class Ultra 
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

        std::string primConstantAttributes(const torch::jit::Node* node, size_t level);
        std::string primConstantAttributesValue(const torch::jit::Node* node, const torch::jit::Symbol& name, size_t level);
        std::string ivalue(const torch::jit::Value* output, const c10::IValue& v, size_t level);
        std::string type(const at::TypePtr& tPtr);
        std::string primConstantTensor(const at::Tensor& t, const torch::jit::Node* node, size_t level);
        std::tuple<std::string, std::string> cppTypeFrom(const at::ScalarType& st);
        std::string constantInitData(const at::Tensor& tensor);
        std::string atNative(torch::jit::Node* node, size_t level);
        std::string atNativeOut(torch::jit::Node* node, size_t level);
        // Some corner case conditions for OP's overloading
        bool extraConditionsOn(torch::jit::Node* node);
        // Handles None = prim::Constant() case
        // Possible outputs are : undefined tensor or c10::nullopt
        void handleNonePrimConstant(torch::jit::Node* node, size_t arg_index);
        // True if all arugments are integral types otherwise False
        bool allArgsScalar(torch::jit::Node* node);

        
        
        void syntheticLib();
        void writeCMakeListsTxt();
        std::string declaration();
        std::string returns();
        void setupWorkspace() const;
        c10::ArrayRef<torch::jit::Value*> inputs() const;
        void processAttributes(const torch::jit::Node* node, bool ignore_subgraph = false) const;
        void processAttributeValue(const torch::jit::Node* node, const c10::Symbol& name) const;
        std::string generateCPPForwardCode() const;
        void processReturns() const;
        void processNode(torch::jit::Node* node, std::vector<const torch::jit::Node*>* groups) const;
        void constValueListWithTypes(const c10::ArrayRef<torch::jit::Value*> inputs) const;

    private:
        torch::jit::script::Module module_; // The model
        std::shared_ptr<torch::jit::Graph> graph_; // graph extracted from model
        std::unique_ptr<c10::FunctionSchema> schema_; // forward schema
        std::string declaration_;
        std::string code_; // forward.cpp file's content
        bool first_time_; // used for caching
        std::unordered_set<std::string> global_scope_; // all global declarations/definitions
        // Static members
        static std::unordered_map<std::string, std::string> s_natives_; // map of aten::native ops
        static std::unordered_map<std::string, std::string> s_native_outs_; // map of aten::native ops with 'out' versions
        static int s_phiLoop_id_; // used for generate unique initial loop condition variable name
};

int forward(const std::string& workspace);

} // ultra