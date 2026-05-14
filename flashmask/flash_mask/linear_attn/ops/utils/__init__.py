# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Original portions of this file are licensed under the MIT License.
# See the LICENSE-MIT file or the original project license for details.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from fla/ops/utils/__init__.py for PaddlePaddle

from .cumsum import (
    chunk_global_cumsum,
    chunk_global_cumsum_scalar,
    chunk_global_cumsum_vector,
    chunk_local_cumsum,
    chunk_local_cumsum_scalar,
    chunk_local_cumsum_vector,
)
from .index import (
    get_max_num_splits,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_cu_seqlens_from_lens,
    prepare_cu_seqlens_from_mask,
    prepare_lens,
    prepare_lens_from_mask,
    prepare_position_ids,
    prepare_sequence_ids,
    prepare_token_indices,
)
from .softmax import softmax_bwd, softmax_fwd
from .softplus import softplus
from .solve_tril import solve_tril
