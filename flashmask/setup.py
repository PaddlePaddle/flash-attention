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

from setuptools import find_packages
from paddle.utils.cpp_extension import CUDAExtension, setup

import os
# lib_dir = os.path.abspath('flash_mask/build')
lib_dir = '/usr/local/lib/python3.10/dist-packages/paddle/libs/'

setup(
    name='flash_mask',
    version='4.0',
    packages=find_packages(),
    author='PaddlePaddle',
    description='FlashMask: Efficient and Rich Mask Extension of FlashAttention',
    install_requires=[
        'nvidia-cutlass==4.2.0.0',
        'nvidia-cutlass-dsl==4.3.0',
        'typing_extensions',
    ],
    python_requires='>=3.10',
    ext_modules=[
        CUDAExtension(
            # [BQW_CHANGE] 保持原始名称 'flash_mask'，Paddle 会自动生成 flash_mask_pd_.so
            name='flash_mask',
            sources=[
                'flash_mask/flashmask_attention_v3/csrc/flashmask_v3.cpp',
                'flash_mask/flashmask_attention_v3/csrc/flashmask_v3_kernel.cu',
                # [BQW_CHANGE] 启用 flash_attn_v3_utils.cu 编译 (之前被注释掉导致链接错误)
                'flash_mask/flashmask_attention_v3/csrc/flash_attn_v3_utils.cu',
            ],
            # [BQW_CHANGE] 添加 include 目录，确保头文件能正确找到
            include_dirs=[
                'flash_mask/flashmask_attention_v3/csrc',
                'flash_mask/flashmask_attention_v3',
            ],
            library_dirs=[lib_dir],
            libraries=['flashmaskv2'],
            extra_compile_args={
                'nvcc': ['-gencode', 'arch=compute_90,code=sm_90', '-O3', '-DPADDLE_WITH_FLASHATTN_V3=1'],
                'cxx': ['-O3', '-DPADDLE_WITH_FLASHATTN_V3=1'],
            },
            # [BQW_CHANGE] 添加 rpath，确保运行时能找到 libflashmaskv2.so
            extra_link_args=['-Wl,-rpath,' + lib_dir],
        )
    ]
)
