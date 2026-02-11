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


# [BQW_CHANGE] 在 import 前先加载 flash_mask_pd_.so 并注册自定义算子
# Paddle CUDAExtension 生成 flash_mask_pd_.so，需要手动加载注册
import os
import paddle

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
_so_path = os.path.join(_parent_dir, "flash_mask_pd_.so")

if os.path.exists(_so_path):
    paddle.utils.cpp_extension.load_op_meta_info_and_register_op(_so_path)
else:
    print(f"[WARNING] flash_mask_pd_.so not found at {_so_path}, custom ops may not be available")

from .flashmask_attention_v3.interface import flashmask_attention

__all__ = ["flashmask_attention"]
