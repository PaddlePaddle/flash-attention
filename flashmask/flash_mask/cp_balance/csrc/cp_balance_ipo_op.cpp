// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/extension.h"
#include "cp_balance_fast.hpp"

std::vector<paddle::Tensor> CpBalanceIpoKernel(
    const paddle::Tensor& weights, int M) {
    int N = static_cast<int>(weights.shape()[0]);
    int K = N / M;

    auto assign_out = paddle::empty(
        {static_cast<int64_t>(M), static_cast<int64_t>(K)},
        paddle::DataType::INT32, paddle::CPUPlace());

    int max_load = CpBalanceSolver::solve_to(
        weights.data<int>(), N, M, assign_out.data<int>());

    auto ml_out = paddle::full({1}, max_load, paddle::DataType::INT32);
    return {assign_out, ml_out};
}

PD_BUILD_OP(cp_balance_ipo)
    .Inputs({"Weights"})
    .Attrs({"M: int"})
    .Outputs({"Assign", "MaxLoad"})
    .SetKernelFn(PD_KERNEL(CpBalanceIpoKernel));
