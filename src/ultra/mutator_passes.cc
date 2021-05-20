#include "ultra.h"
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <memory>

namespace ultra {

using namespace torch::jit;

void Ultra::try_to_use_inplace_relu()
{
    // Currently only top level block is supported 
    // but in general we should consider all embedding 
    // blocks (this is currently not critical because 
    // most of time either there is only one level or 
    // 99+% of work is done in first level).
    auto block = graph_ -> block();
    for (auto it = block -> nodes().begin(); it != block -> nodes().end(); ++it) 
    {
        auto* node = *it;
        if (std::string(node -> kind() . toQualString()) == "aten::relu" and 
            node -> input() -> uses() . size() == 1)
        {
            node -> output() -> replaceAllUsesWith(node -> input());
            TORCH_CHECK(inplace_nodes_.count(node) == 0, "Inplace node already in set.");
            inplace_nodes_.insert(node);
        }

        if (std::string(node -> kind() . toQualString()) == "aten::dropout" and
            node -> inputs()[0] -> uses() . size() == 1)
        {
            node -> output() -> replaceAllUsesWith(node -> inputs()[0]);
            TORCH_CHECK(inplace_nodes_.count(node) == 0, "Inplace node already in set.");
            inplace_nodes_.insert(node);
        }
    }
}

} // namespace ultra