#pragma once

#include <torch/script.h>

namespace ultra 
{

    // Checks if given schema is contains in NativeSchemas library
    bool containsInNativeLibrary(const c10::FunctionSchema& schema);

    // Checks if gyven schema contains 'out' overloading
    std::string getOutVariant(const c10::FunctionSchema& schema);

    // Get schema's native name
    std::string getNativeVariant(const c10::FunctionSchema& schema);
} // namespace ultra