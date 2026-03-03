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


# [BQW_CHANGE] 在 import 前先加载 flash_mask.so 并注册自定义算子
# Paddle CUDAExtension 生成 flash_mask.so，需要手动加载注册
import os
import paddle

# 尝试加载编译好的 so 文件
# 1. 首先尝试从 site-packages 加载（pip install 后）
# 2. 然后尝试从本地 build 目录加载（python setup.py install 后）
_so_loaded = False

# 尝试从已安装的模块中加载
try:
    import flash_mask as _flash_mask_module
    _so_path = _flash_mask_module.__file__
    if _so_path.endswith('.so'):
        paddle.utils.cpp_extension.load_op_meta_info_and_register_op(_so_path)
        _so_loaded = True
except Exception:
    pass

# 如果还没加载，尝试从 build 目录加载
if not _so_loaded:
    _curr_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_curr_dir)

    # 检查可能的 so 文件位置
    _possible_paths = [
        os.path.join(_parent_dir, "build", "flash_mask", "lib.linux-x86_64-cpython-310", "flash_mask.so"),
        os.path.join(_parent_dir, "flash_mask.so"),
    ]

    for _so_path in _possible_paths:
        if os.path.exists(_so_path):
            paddle.utils.cpp_extension.load_op_meta_info_and_register_op(_so_path)
            _so_loaded = True
            break

if not _so_loaded:
    print("[WARNING] flash_mask.so not found, custom ops may not be available")

from .flashmask_attention_v3.interface import flashmask_attention

__all__ = ["flashmask_attention"]
