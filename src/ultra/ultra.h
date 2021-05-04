#pragma once

#include <torch/script.h>
#include <glog/logging.h>
#include <unordered_map>
#include <filesystem>

namespace ultra 
{

namespace fs = std::filesystem;

class Ultra 
{
    public:
        explicit Ultra(torch::jit::script::Module module);
        ~Ultra() = default;
        Ultra(const Ultra&) = delete;
        Ultra(Ultra&&) = delete;
        Ultra& operator=(const Ultra&) = delete;
        Ultra& operator=(Ultra&&) = delete;

    public:
        void buildLibrary(const fs::path& w);

    private:
        void OptimizeGraph();
        void RemoveSelfFromGraphInput();

        void writeHPP(const fs::path& w);
        void writeCPP(const fs::path& w);
        void graphTraversal();
        std::string blockTraversal(torch::jit::Block* block, size_t level);
        std::string primNode(torch::jit::Node* node, size_t level);
        std::string phiNode(torch::jit::Node* node, size_t level);
        std::string normalizeName(const std::string& name) const;
        std::string primConstantAttributes(const torch::jit::Node* node, size_t level);
        std::string primConstantAttributesValue(const torch::jit::Node* node, const torch::jit::Symbol& name, size_t level);
        std::string ivalue(const torch::jit::Value* output, const c10::IValue& v, size_t level);
        std::string type(const at::TypePtr& tPtr);
        std::string primConstantTensor(const at::Tensor& t, const torch::jit::Node* node, size_t level);
        std::tuple<std::string, std::string> cppTypeFrom(const at::ScalarType& st);
        std::string constantInitData(const at::Tensor& tensor);
        std::string atNative(torch::jit::Node* node, size_t level);
        std::string atNativeOut(torch::jit::Node* node, size_t level);
        void foldSizeLenGT();
        void foldDimNE();
        void foldDimEQ();

        
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
        torch::jit::script::Module module_;
        std::shared_ptr<torch::jit::Graph> graph_;
        std::unique_ptr<c10::FunctionSchema> schema_;
        std::string declaration_;
        std::string code_;
        bool first_time_;
        std::unordered_set<std::string> global_scope_;

        static std::unordered_map<std::string, std::string> s_natives_;
        static std::unordered_map<std::string, std::string> s_native_outs_;
        static int s_phiLoop_id_;
};

int forward(const std::string& workspace);

} // ultra