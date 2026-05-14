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

from .chunk import chunk_gated_delta_rule, chunk_gdn
from .fused_recurrent import fused_recurrent_gated_delta_rule, fused_recurrent_gdn
from .naive import naive_recurrent_gated_delta_rule

__all__ = [
    "chunk_gated_delta_rule", "chunk_gdn",
    "fused_recurrent_gated_delta_rule", "fused_recurrent_gdn",
    "naive_recurrent_gated_delta_rule",
]
