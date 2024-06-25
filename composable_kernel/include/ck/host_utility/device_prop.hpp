// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <map>
#include <hip/hip_runtime.h>

namespace ck {

inline std::string get_device_name()
{
    
    return "gfx940";
}

inline bool is_xdl_supported()
{
    return true;
}

} // namespace ck
