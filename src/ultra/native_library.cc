#include "native_library.h"
#include "native_functions.h"

namespace ultra {

using namespace torch::jit;

// Encapsulates native functions schemas and 
// corresponding CPU dispatcher functions names 
// if exists
struct NativeSchemas final
{
    NativeSchemas()
    {
        native_schemas_.reserve(n_functions.size());
        TORCH_CHECK(n_functions.size() % 2 == 0);
        for (auto i = 0; i < n_functions.size(); i += 2)
        {
            c10::FunctionSchema schema = torch::schema(n_functions[i].c_str());
            schema.setNamespaceIfNotSet("aten");
            native_schemas_.emplace_back(std::move(
                std::make_pair(
                    schema,
                    c10::OperatorName(n_functions[i + 1], "")
                )
            ));
        }
    }

    bool contains(const c10::FunctionSchema& schema) const
    {
        for (auto& el: native_schemas_)
        {
            if (el.first == schema) 
            {
                return true;
            }
        }
        return false;
    }

    std::string getOutVariant(const c10::FunctionSchema& schema) const
    {
        for (auto& el: native_schemas_)
        {
            if (el.first.name() == schema.name() and el.first.overload_name() == "out")
            {
                return el.second.name == "NO_NS::NO_FUNCTION" ? schema.name() + "_out" : el.second.name;
                return el.second.name == "NO_NS::NO_FUNCTION" ? "" : el.second.name;
            }
        }
        std::cerr << " >>> NO OUT VARIANT FOR " << schema << std::endl;
        return "";
    }

    std::string getNativeVariant(const c10::FunctionSchema& schema) const 
    {
        for (auto& el: native_schemas_)
        {
            if (el.first.name() == schema.name() and el.first.overload_name() != "out")
            {
                return el.second.name == "NO_NS::NO_FUNCTION" ? schema . name() : el.second.name;
            }
        }
        std::cerr << " >>> NO NATIVE NAME FOR " << schema << std::endl;
        return "";
    }
    // TODO make c10::FunctionSchema hashable
    // {<Native function schema>, <CPU dispatcher function name/"NO_FUNCTION">}
    std::vector<std::pair<c10::FunctionSchema, c10::OperatorName>> native_schemas_;
};

// Load Native functions schema's , which will help us
// to check existence of different variants of IR instruction
NativeSchemas& getNative()
{
    static NativeSchemas ns;
    return ns;
}

bool containsInNativeLibrary(const c10::FunctionSchema& schema)
{
    return getNative().contains(schema);
}

std::string getOutVariant(const c10::FunctionSchema& schema)
{
    return getNative().getOutVariant(schema);
}

std::string getNativeVariant(const c10::FunctionSchema& schema)
{
    return getNative().getNativeVariant(schema);
}

} // namespace ultra