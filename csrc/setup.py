#licensed under the Apache License, Version 2.0 (the "License");
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
import multiprocessing
from pathlib import Path
import os

def get_gencode_flags():
    import paddle

    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]


def run(func):
    p = multiprocessing.Process(target=func)
    p.start()
    p.join()


def change_pwd():
    path = os.path.dirname(__file__)
    if path:
        os.chdir(path)


this_dir = os.path.dirname(os.path.abspath(__file__))
def setup_fused_ln():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    gencode_flags = get_gencode_flags()
    change_pwd()
    setup(
        name="flash_attn",
        ext_modules=CUDAExtension(
            sources=[
                "flash_attn.cu",
     "flash_attn/src/cuda_utils.cu",
     "flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim224_fp16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim224_bf16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu",
     "flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim224_fp16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim224_bf16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu",
     "flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + gencode_flags,
                "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        "--ptxas-options=-v",
                        # "--ptxas-options=-O2",
                        "-lineinfo"
                ]
            },
            include_dirs=[
             "flash_attn",
             "flash_attn/src",
             "cutlass/include",
            ],
        ),
    )


run(setup_fused_ln)
