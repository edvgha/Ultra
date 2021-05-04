#include "ultra.h"
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <memory>

namespace ultra {

using namespace torch::jit;

void Ultra::foldSizeLenGT()
{

    auto registrar = c10::RegisterOperators().op("Ultra::size_len_gt", 
                                                 c10::RegisterOperators::options().kernel(at::DispatchKey::CPU, 
                                                 [] (at::Tensor&, int64_t& i) -> bool {return true;}));

    auto op = c10::Dispatcher::singleton().findSchema({"Ultra::size_len_gt", ""});
    TORCH_CHECK(op.has_value(), "Failed to register Ultra::size_len_gt");

    std::string patter = R"IR(
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
    folder.RegisterRewritePattern(patter, target);
    folder.runOnGraph(graph_);
}

void Ultra::foldDimNE()
{

    auto registrar = c10::RegisterOperators().op("Ultra::dim_ne", 
                                                 c10::RegisterOperators::options().kernel(at::DispatchKey::CPU, 
                                                 [] (at::Tensor&, int64_t& i) -> bool {return true;}));

    auto op = c10::Dispatcher::singleton().findSchema({"Ultra::dim_ne", ""});
    TORCH_CHECK(op.has_value(), "Failed to register Ultra::dim_ne");

    std::string patter = R"IR(
        graph(%x, %y):
            %a : int = aten::dim(%x)
            %d : bool = aten::ne(%a, %y)
            return (%d))IR";
    std::string target = R"IR(
        graph(%x, %y):
            %d : bool = Ultra::dim_ne(%x, %y)
            return (%d))IR";
    SubgraphRewriter folder;
    folder.RegisterRewritePattern(patter, target);
    folder.runOnGraph(graph_);
}

void Ultra::foldDimEQ()
{

    auto registrar = c10::RegisterOperators().op("Ultra::dim_eq", 
                                                 c10::RegisterOperators::options().kernel(at::DispatchKey::CPU, 
                                                 [] (at::Tensor&, int64_t& i) -> bool {return true;}));

    auto op = c10::Dispatcher::singleton().findSchema({"Ultra::dim_eq", ""});
    TORCH_CHECK(op.has_value(), "Failed to register Ultra::dim_eq");

    std::string patter = R"IR(
        graph(%x, %y):
            %a : int = aten::dim(%x)
            %d : bool = aten::eq(%a, %y)
            return (%d))IR";
    std::string target = R"IR(
        graph(%x, %y):
            %d : bool = Ultra::dim_eq(%x, %y)
            return (%d))IR";
    SubgraphRewriter folder;
    folder.RegisterRewritePattern(patter, target);
    folder.runOnGraph(graph_);
}

} // namespace ultra