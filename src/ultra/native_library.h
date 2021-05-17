#pragma once

#include <torch/script.h>

namespace ultra 
{

    // TODO
    bool containsInNativeLibrary(const c10::FunctionSchema& schema);
} // namespace ultra