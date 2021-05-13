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

bool Ultra::isPrint(char s) 
{
  return s > 0x1f && s < 0x7f;
}

std::string Ultra::printQuotedString(const std::string& str) 
{
    std::ostringstream oss;
    oss << "\"";
    for (auto s : str) 
    {
        switch (s) 
        {
            case '\\':
                oss << "\\\\";
                break;
            case '\'':
                oss << "\\'";
                break;
            case '\"':
                oss << "\\\"";
                break;
            case '\a':
                oss << "\\a";
                break;
            case '\b':
                oss << "\\b";
                break;
            case '\f':
                oss << "\\f";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            case '\v':
                oss << "\\v";
                break;
            default:
                if (isPrint(s)) 
                {
                    oss << s;
                } 
                else 
                {
                    // C++ io has stateful formatting settings. Messing with
                    // them is probably worse than doing this manually.
                    char buf[4] = "000";
                    buf[2] += s % 8;
                    s /= 8;
                    buf[1] += s % 8;
                    s /= 8;
                    buf[0] += s;
                    oss << "\\" << buf;
                }
                break;
        }
    }
    oss << "\"";
    return oss.str();
}

std::string Ultra::nodeArgument(const c10::Argument& arg)
{
    std::ostringstream oss;
    // for adjusting the ? position.
    // in schema, we have Tensor?(a!) input, and t(a!)?.
    // however, t?(a!) doesn't work with schema parser.
    // so we always use Type(alias)? format
    auto type = arg.type();
    bool is_opt = type -> kind() == OptionalType::Kind;
    // auto unopt_type = is_opt ? type -> castRaw<OptionalType>() -> getElementType() : type;

    auto unopt_type = type;
    if (is_opt) 
    {
        unopt_type = static_cast<const OptionalType*>(&(*type))->getElementType();
    }

    if (unopt_type -> kind() == ListType::Kind && arg.N()) 
    {
        // sized lists get size N from arg, not type
        auto list = unopt_type -> cast<c10::ListType>();
        oss << list -> getElementType() -> str() << "[" << *arg.N() << "]";
    } 
    else 
    {
        oss << unopt_type -> str();
    }

    // Alias info currently not used
    // if (arg.alias_info()) oss << arg.alias_info().value();

    if (is_opt) 
    {
        oss << "?";
    }

    if (not arg.name().empty()) 
    {
        oss << " " << arg.name();
    }

    if (arg.default_value()) 
    {
        oss << "=";
        if (type -> kind() == c10::TypeKind::StringType or (unopt_type -> kind() == c10::TypeKind::StringType and not arg.default_value().value().isNone())) {
            oss << printQuotedString(arg.default_value().value().toStringRef());
        } else {
            oss << arg.default_value().value(); // +
        }
    }
    return oss.str();
}

std::string Ultra::nodeSchema(Node* node, size_t level)
{
    std::ostringstream oss;
    const c10::FunctionSchema* schema = node -> maybeSchema();
    if (not schema) 
    {
        return "";
    }
    std::cerr << "---------------------" << std::endl;
    node -> dump();
    std::cerr << "+++++++++++++++++++++" << std::endl;

    oss << schema -> name();
    if (schema -> overload_name() != "") {
        oss << "." << schema -> overload_name();
    }
    oss << "(";

    bool seen_kwarg_only = false;
    for(size_t i = 0; i < schema -> arguments().size(); ++i) 
    {
        if (i > 0) 
        {
            oss << ", ";
        }
        if (schema -> arguments()[i].kwarg_only() && !seen_kwarg_only) 
        {
            oss << "*, ";
            seen_kwarg_only = true;
        }
        oss << nodeArgument(schema -> arguments()[i]);
        // oss << schema -> arguments()[i];
    }

    if(schema -> is_vararg()) 
    {
        if(schema -> arguments().size() > 0)
        {
            oss << ", ";
        }
        oss << "...";
    }

    oss << ") -> ";

    const auto& returns = schema -> returns();
    oss << "(";
    for(size_t i = 0; i < returns.size(); ++i) 
    {
        if (i > 0) 
        {
            oss << ", ";
        }
        oss << returns.at(i);
    }

    if (schema -> is_varret()) 
    {
        if (returns.size() != 0) 
        {
            oss << ", ";
        }
        oss << "...";
    }
    oss << ")";
    return oss.str();
}

} // namespace ultra