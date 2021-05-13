#include "ultra.h"
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <memory>

namespace ultra {

using namespace torch::jit;

// Register cusom Operators
TORCH_LIBRARY(Ultra, m) {
  m.def("size_len_gt(Tensor self, int i) -> bool");
  m.def("dim_ne(Tensor self, int i) -> bool");
  m.def("dim_eq(Tensor self, int i) -> bool");
  m.def("dim(Tensor self) -> int");
  m.def("ne(int a, int b) -> bool");
}

void Ultra::ne()
{
    std::string pattern = R"IR(
        graph(%x, %y):
            %c = aten::ne(%x, %y)
            return (%c))IR";
    std::string target = R"IR(
        graph(%x, %y):
            %c = Ultra::ne(%x, %y)
            return (%c))IR";
    SubgraphRewriter folder;
    folder.RegisterRewritePattern(pattern, target);
    folder.runOnGraph(graph_);
}

void Ultra::dimToSizes()
{
    std::string pattern = R"IR(
        graph(%x):
            %c : int = aten::dim(%x)
            return (%c))IR";
    std::string target = R"IR(
        graph(%x):
            %c : int = Ultra::dim(%x)
            return (%c))IR";
    SubgraphRewriter folder;
    folder.RegisterRewritePattern(pattern, target);
    folder.runOnGraph(graph_);
}

void Ultra::foldSizeLenGT()
{
    std::string pattern = R"IR(
        graph(%x, %y):
            %b : int[] = aten::size(%x)
            %c : int = aten::len(%b)
            %d : bool = aten::gt(%c, %y)
            return (%d))IR";
    std::string target = R"IR(
        graph(%x, %y):
            %d : bool = Ultra::size_len_gt(%x, %y)
            return (%d))IR";
    SubgraphRewriter folder;
    folder.RegisterRewritePattern(pattern, target);
    folder.runOnGraph(graph_);
}

void Ultra::foldDimNE()
{
    std::string pattern = R"IR(
        graph(%x, %y):
            %a : int = aten::dim(%x)
            %d : bool = aten::ne(%a, %y)
            return (%d))IR";
    std::string target = R"IR(
        graph(%x, %y):
            %d : bool = Ultra::dim_ne(%x, %y)
            return (%d))IR";
    SubgraphRewriter folder;
    folder.RegisterRewritePattern(pattern, target);
    folder.runOnGraph(graph_);
}

void Ultra::foldDimEQ()
{
    std::string pattern = R"IR(
        graph(%x, %y):
            %a : int = aten::dim(%x)
            %d : bool = aten::eq(%a, %y)
            return (%d))IR";
    std::string target = R"IR(
        graph(%x, %y):
            %d : bool = Ultra::dim_eq(%x, %y)
            return (%d))IR";
    SubgraphRewriter folder;
    folder.RegisterRewritePattern(pattern, target);
    folder.runOnGraph(graph_);
}

} // namespace ultra