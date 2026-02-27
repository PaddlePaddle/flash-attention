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

# 整合 cmake 构建流程到 setup.py，pip install . 一步到位
# 不再需要手动 mkdir build && cmake .. && make
#
# 使用方法:
#   pip install .                        # 标准安装（自动 cmake + 编译）
#   pip install -e . --no-build-isolation  # 开发安装
#
# 环境变量:
#   FLASH_MASK_SKIP_CMAKE=1    跳过 cmake（假设 libflashmaskv3.so 已存在）
#   FLASH_MASK_FORCE_REBUILD=1 强制重新 cmake
#   FLASH_MASK_CMAKE_ARGS      额外 cmake 参数（空格分隔）
#   FLASH_MASK_LIB_DIR         手动指定 libflashmaskv2/v3.so 所在目录

import os
import sys
import subprocess
import shutil

from setuptools import find_packages
from paddle.utils.cpp_extension import CUDAExtension, setup

# ============================================================
# 配置区
# ============================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FLASH_MASK_DIR = os.path.join(ROOT_DIR, 'flash_mask')
BUILD_DIR = os.path.join(FLASH_MASK_DIR, 'build')
# libflashmaskv3.so 安装到 flash_mask/lib/ 下，随 package_data 分发
INSTALL_LIB_DIR = os.path.join(FLASH_MASK_DIR, 'lib')

SKIP_CMAKE = os.environ.get('FLASH_MASK_SKIP_CMAKE', '0') == '1'
FORCE_REBUILD = os.environ.get('FLASH_MASK_FORCE_REBUILD', '0') == '1'
EXTRA_CMAKE_ARGS = os.environ.get('FLASH_MASK_CMAKE_ARGS', '').split()
MANUAL_LIB_DIR = os.environ.get('FLASH_MASK_LIB_DIR', '')

# ============================================================
# Step 1: 构建 libflashmaskv3.so (底层 CUDA kernel 库)
# ============================================================
LIB_NAME = 'flashmaskv3'
LIB_FILE = f'lib{LIB_NAME}.so'

def find_or_build_lib():
    """找到或构建 libflashmaskv3.so，返回 (lib_dir, lib_name)"""
    global LIB_NAME

    if MANUAL_LIB_DIR:
        lib_dir = os.path.abspath(MANUAL_LIB_DIR)
        if os.path.exists(os.path.join(lib_dir, 'libflashmaskv3.so')):
            return lib_dir, 'flashmaskv3'
        else:
            print(f"[WARNING] No flashmask lib found in {lib_dir}")
            return lib_dir, 'flashmaskv3'

    if SKIP_CMAKE:
        return BUILD_DIR, 'flashmaskv3'

    # 自动 cmake 构建
    lib_so_path = os.path.join(BUILD_DIR, LIB_FILE)
    need_build = FORCE_REBUILD or not os.path.exists(lib_so_path)

    if need_build:
        print("=" * 60)
        print(f"Building {LIB_FILE} via cmake...")
        print("=" * 60)

        os.makedirs(BUILD_DIR, exist_ok=True)
        cmake_args = [
            'cmake', '..',
            '-DWITH_FLASHATTN_V3=ON',
            '-DDISABLE_FLASHMASK_V3_BACKWARD=OFF',
        ] + EXTRA_CMAKE_ARGS

        print(f"  cmake args: {' '.join(cmake_args)}")
        subprocess.check_call(cmake_args, cwd=BUILD_DIR)

        nproc = os.cpu_count() or 4
        print(f"  make -j{nproc}")
        subprocess.check_call(['make', f'-j{nproc}'], cwd=BUILD_DIR)

        if not os.path.exists(lib_so_path):
            raise RuntimeError(f"cmake build completed but {LIB_FILE} not found at {lib_so_path}")
        print(f"  {LIB_FILE} built successfully")
    else:
        print(f"  {LIB_FILE} already exists, skipping cmake (set FLASH_MASK_FORCE_REBUILD=1 to force)")

    return BUILD_DIR, 'flashmaskv3'

LIB_DIR, LIB_NAME = find_or_build_lib()
LIB_DIR = os.path.abspath(LIB_DIR)
LIB_FILE = f'lib{LIB_NAME}.so'

# 将 libflashmaskv3.so 拷贝到 flash_mask/lib/ 下
os.makedirs(INSTALL_LIB_DIR, exist_ok=True)
src_lib = os.path.join(LIB_DIR, LIB_FILE)
dst_lib = os.path.join(INSTALL_LIB_DIR, LIB_FILE)
if os.path.exists(src_lib) and (not os.path.exists(dst_lib) or
        os.path.getmtime(src_lib) > os.path.getmtime(dst_lib)):
    shutil.copy2(src_lib, dst_lib)
    print(f"  Copied {LIB_FILE} -> flash_mask/lib/")

# ============================================================
# Step 2: 构建自定义算子
# ============================================================
setup(
    name='flash_mask',
    version='4.0',
    packages=find_packages(),
    package_data={
        'flash_mask': ['lib/*.so'],
    },
    author='PaddlePaddle',
    description='FlashMask: Efficient and Rich Mask Extension of FlashAttention',
    install_requires=[
        'typing_extensions',
        'nvidia-cutlass==4.2.0.0',
        'nvidia-cutlass-dsl==4.3.0',
    ],
    python_requires='>=3.10',
    ext_modules=[
        CUDAExtension(
            name='flash_mask_package',
            sources=[
                'flash_mask/flashmask_attention_v3/csrc/flashmask_v3.cpp',
                'flash_mask/flashmask_attention_v3/csrc/flashmask_v3_kernel.cu',
                'flash_mask/flashmask_attention_v3/csrc/flashmask_v3_grad_kernel.cu',
                'flash_mask/flashmask_attention_v3/csrc/flash_attn_v3_utils.cu',
            ],
            include_dirs=[
                'flash_mask/flashmask_attention_v3/csrc',
                'flash_mask/flashmask_attention_v3',
            ],
            library_dirs=[LIB_DIR, INSTALL_LIB_DIR],
            libraries=[LIB_NAME],
            extra_compile_args={
                'nvcc': [
                    '-gencode', 'arch=compute_90,code=sm_90',
                    '-O3',
                    '-DPADDLE_WITH_FLASHATTN_V3=1',
                    '-std=c++17',
                ],
                'cxx': [
                    '-O3',
                    '-DPADDLE_WITH_FLASHATTN_V3=1',
                    '-std=c++17'],
            },
            extra_link_args=[
                '-Wl,-rpath,$ORIGIN/flash_mask/lib',
                '-Wl,-rpath,$ORIGIN',
                f'-Wl,-rpath,{INSTALL_LIB_DIR}',
                f'-Wl,-rpath,{LIB_DIR}',
            ],
        )
    ]
)
