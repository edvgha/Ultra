#pragma once

#include <torch/script.h>

namespace ultra 
{

    // Checks if given schema is contains in NativeSchemas library
    bool containsInNativeLibrary(const c10::FunctionSchema& schema);
} // namespace ultra