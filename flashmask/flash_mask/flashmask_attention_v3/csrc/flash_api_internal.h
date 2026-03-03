/******************************************************************************
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

// Forward declarations for the core flash attention functions defined in flash_api.cu.
// This allows the Paddle adapter code to call them directly without going through
// the C ABI wrapper layer (eliminating the need for a separate libflashmaskv3.so).

#pragma once

#include "../flash.h"
#include <cuda_runtime.h>

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream);
void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream);
void run_mha_fwd_combine(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);
bool get_pagedkv_tma(Flash_fwd_params const& params);
bool get_pack_gqa(Flash_fwd_params const& params);
int get_num_splits(Flash_fwd_params const& params);
