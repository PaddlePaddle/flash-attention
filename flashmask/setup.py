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

# Single-step build: compiles everything (CUDA kernels + Paddle custom ops)
# into one flash_mask_pd_.so. No separate cmake / libflashmaskv3.so needed.
#
# Usage:
#   python setup.py install
#   pip install .
#   pip install -e . --no-build-isolation

import os
import sys
import subprocess

from setuptools import find_packages
from paddle.utils.cpp_extension import CUDAExtension, setup

# ============================================================
# Config
# ============================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FLASH_MASK_DIR = os.path.join(ROOT_DIR, 'flash_mask')
FA_V3_DIR = os.path.join(FLASH_MASK_DIR, 'flashmask_attention_v3')
INST_DIR = os.path.join(FA_V3_DIR, 'instantiations')

# Feature toggles (match CMakeLists.txt defaults)
DISABLE_FP16      = os.environ.get('DISABLE_FLASHMASK_V3_FP16', '0') == '1'
DISABLE_FP8       = os.environ.get('DISABLE_FLASHMASK_V3_FP8', '1') == '1'
DISABLE_HDIM64    = os.environ.get('DISABLE_FLASHMASK_V3_HDIM64', '0') == '1'
DISABLE_HDIM96    = os.environ.get('DISABLE_FLASHMASK_V3_HDIM96', '1') == '1'
DISABLE_HDIM128   = os.environ.get('DISABLE_FLASHMASK_V3_HDIM128', '0') == '1'
DISABLE_HDIM192   = os.environ.get('DISABLE_FLASHMASK_V3_HDIM192', '1') == '1'
DISABLE_HDIM256   = os.environ.get('DISABLE_FLASHMASK_V3_HDIM256', '0') == '1'
DISABLE_SPLIT     = os.environ.get('DISABLE_FLASHMASK_V3_SPLIT', '1') == '1'
DISABLE_PAGEDKV   = os.environ.get('DISABLE_FLASHMASK_V3_PAGEDKV', '1') == '1'
DISABLE_SOFTCAP   = os.environ.get('DISABLE_FLASHMASK_V3_SOFTCAP', '1') == '1'
DISABLE_PACKGQA   = os.environ.get('DISABLE_FLASHMASK_V3_PACKGQA', '1') == '1'
DISABLE_BACKWARD  = os.environ.get('DISABLE_FLASHMASK_V3_BACKWARD', '0') == '1'
DISABLE_SM8X      = os.environ.get('DISABLE_FLASHMASK_V3_SM8X', '1') == '1'

# ============================================================
# Step 1: Ensure instantiation .cu files are generated
# ============================================================
if not os.path.isdir(INST_DIR) or len(os.listdir(INST_DIR)) == 0:
    print("Generating kernel instantiation files...")
    subprocess.check_call(
        [sys.executable, os.path.join(FA_V3_DIR, 'generate_kernels.py'),
         '-o', INST_DIR],
        cwd=FLASH_MASK_DIR,
    )

# ============================================================
# Step 2: Collect source files (matching CMakeLists.txt logic)
# ============================================================

# Enabled head dimensions
hdims = []
if not DISABLE_HDIM64:  hdims.append('64')
if not DISABLE_HDIM96:  hdims.append('96')
if not DISABLE_HDIM128: hdims.append('128')
if not DISABLE_HDIM192: hdims.append('192')
if not DISABLE_HDIM256: hdims.append('256')

# --- Forward SM90 ---
dtypes_fwd_sm90 = ['bf16']
if not DISABLE_FP16: dtypes_fwd_sm90.append('fp16')
if not DISABLE_FP8:  dtypes_fwd_sm90.append('e4m3')

split_suffixes = ['']
if not DISABLE_SPLIT: split_suffixes.append('_split')

paged_suffixes = ['']
if not DISABLE_PAGEDKV: paged_suffixes.append('_paged')

softcap_fwd_suffixes = ['']
if not DISABLE_SOFTCAP: softcap_fwd_suffixes.append('_softcap')

# For softcap "all": if softcap disabled, use empty; if enabled, use _softcapall
softcap_all_suffixes = [''] if DISABLE_SOFTCAP else ['_softcapall']

packgqa_suffixes = ['']
if not DISABLE_PACKGQA: packgqa_suffixes.append('_packgqa')

instantiation_sources = []

for hdim in hdims:
    for dtype in dtypes_fwd_sm90:
        for split in split_suffixes:
            for paged in paged_suffixes:
                for softcap in softcap_fwd_suffixes:
                    for packgqa in packgqa_suffixes:
                        # Only allow packgqa if not paged and not split (CMake logic)
                        if packgqa == '_packgqa' and (paged != '' or split != ''):
                            continue
                        fname = f'flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}{packgqa}_sm90.cu'
                        fpath = os.path.join(INST_DIR, fname)
                        if os.path.exists(fpath):
                            instantiation_sources.append(fpath)

# --- Forward SM80 ---
if not DISABLE_SM8X:
    dtypes_fwd_sm80 = ['bf16']
    if not DISABLE_FP16: dtypes_fwd_sm80.append('fp16')
    for hdim in hdims:
        for dtype in dtypes_fwd_sm80:
            for split in split_suffixes:
                for paged in paged_suffixes:
                    for softcap in softcap_all_suffixes:
                        fname = f'flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}_sm80.cu'
                        fpath = os.path.join(INST_DIR, fname)
                        if os.path.exists(fpath):
                            instantiation_sources.append(fpath)

# --- Backward SM90 ---
if not DISABLE_BACKWARD:
    dtypes_bwd = ['bf16']
    if not DISABLE_FP16: dtypes_bwd.append('fp16')

    softcap_bwd_all = [''] if DISABLE_SOFTCAP else ['_softcapall']

    for hdim in hdims:
        for dtype in dtypes_bwd:
            for causal in ['', '_causal']:
                for determ in ['', '_determ']:
                    for softcap in softcap_bwd_all:
                        fname = f'flash_bwd_hdim{hdim}_{dtype}{causal}{determ}{softcap}_sm90.cu'
                        fpath = os.path.join(INST_DIR, fname)
                        if os.path.exists(fpath):
                            instantiation_sources.append(fpath)

# --- Backward SM80 ---
if not DISABLE_BACKWARD and not DISABLE_SM8X:
    softcap_bwd_sm80 = ['']
    if not DISABLE_SOFTCAP: softcap_bwd_sm80.append('_softcap')

    for hdim in hdims:
        for dtype in dtypes_bwd:
            for softcap in softcap_bwd_sm80:
                fname = f'flash_bwd_hdim{hdim}_{dtype}{softcap}_sm80.cu'
                fpath = os.path.join(INST_DIR, fname)
                if os.path.exists(fpath):
                    instantiation_sources.append(fpath)

# Core CUDA sources
core_sources = [
    os.path.join(FA_V3_DIR, 'flash_api.cu'),
    os.path.join(FA_V3_DIR, 'flash_prepare_scheduler.cu'),
]
if not DISABLE_SPLIT:
    core_sources.append(os.path.join(FA_V3_DIR, 'flash_fwd_combine.cu'))

# Paddle adapter sources
adapter_sources = [
    'flash_mask/flashmask_attention_v3/csrc/flashmask_v3.cpp',
    'flash_mask/flashmask_attention_v3/csrc/flashmask_v3_kernel.cu',
    'flash_mask/flashmask_attention_v3/csrc/flashmask_v3_grad_kernel.cu',
    'flash_mask/flashmask_attention_v3/csrc/flash_attn_v3_utils.cu',
]

all_sources = adapter_sources + core_sources + instantiation_sources
# Make paths relative to ROOT_DIR for consistency
all_sources = [os.path.relpath(s, ROOT_DIR) if os.path.isabs(s) else s for s in all_sources]

print(f"Total CUDA sources: {len(all_sources)} "
      f"(adapter: {len(adapter_sources)}, core: {len(core_sources)}, "
      f"instantiations: {len(instantiation_sources)})")

# ============================================================
# Step 3: Build compile flags
# ============================================================
disable_defines = []
if DISABLE_FP16:     disable_defines.append('-DFLASHMASK_V3_DISABLE_FP16')
if DISABLE_FP8:      disable_defines.append('-DFLASHMASK_V3_DISABLE_FP8')
if DISABLE_HDIM64:   disable_defines.append('-DFLASHMASK_V3_DISABLE_HDIM64')
if DISABLE_HDIM96:   disable_defines.append('-DFLASHMASK_V3_DISABLE_HDIM96')
if DISABLE_HDIM128:  disable_defines.append('-DFLASHMASK_V3_DISABLE_HDIM128')
if DISABLE_HDIM192:  disable_defines.append('-DFLASHMASK_V3_DISABLE_HDIM192')
if DISABLE_HDIM256:  disable_defines.append('-DFLASHMASK_V3_DISABLE_HDIM256')
if DISABLE_SPLIT:    disable_defines.append('-DFLASHMASK_V3_DISABLE_SPLIT')
if DISABLE_PAGEDKV:  disable_defines.append('-DFLASHMASK_V3_DISABLE_PAGEDKV')
if DISABLE_SOFTCAP:  disable_defines.append('-DFLASHMASK_V3_DISABLE_SOFTCAP')
if DISABLE_PACKGQA:  disable_defines.append('-DFLASHMASK_V3_DISABLE_PACKGQA')
if DISABLE_BACKWARD: disable_defines.append('-DFLASHMASK_V3_DISABLE_BACKWARD')
if DISABLE_SM8X:     disable_defines.append('-DFLASHMASK_V3_DISABLE_SM8X')

nvcc_flags = [
    '-gencode', 'arch=compute_90a,code=sm_90a',
    '-O3',
    '-std=c++17',
    '-DPADDLE_WITH_FLASHATTN_V3=1',
    '-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED',
    '-DCUTLASS_ENABLE_GDC_FOR_SM90',
    '-DCUTLASS_DEBUG_TRACE_LEVEL=0',
    '-DNDEBUG',
    '--use_fast_math',
    '--expt-relaxed-constexpr',
    '-Xcompiler=-fPIC',
    '-Xcompiler=-O3',
    '--ftemplate-backtrace-limit=0',
    '--resource-usage',
    '-lineinfo',
] + disable_defines

cxx_flags = [
    '-O3',
    '-DPADDLE_WITH_FLASHATTN_V3=1',
    '-std=c++17',
] + disable_defines

# ============================================================
# Step 4: Build
# ============================================================
setup(
    name='flash_mask',
    version='4.0',
    packages=find_packages(),
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
            name='flash_mask',
            sources=all_sources,
            include_dirs=[
                'flash_mask/flashmask_attention_v3/csrc',
                'flash_mask/flashmask_attention_v3',
                'flash_mask/flashmask_attention_v3/cutlass/include',
            ],
            extra_compile_args={
                'nvcc': nvcc_flags,
                'cxx': cxx_flags,
            },
        )
    ]
)
