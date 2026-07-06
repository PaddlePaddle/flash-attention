import os
import re
import shutil
import subprocess


def change_pwd():
    path = os.path.dirname(__file__)
    if path:
        os.chdir(path)


def get_cuda_version():
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise FileNotFoundError("nvcc command not found. Please make sure CUDA toolkit is installed and nvcc is in PATH.")
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
    match = re.search(r"release (\d+)\.(\d+)", result.stdout)
    if not match:
        raise ValueError(f"Cannot parse CUDA version from nvcc output:\n{result.stdout}")
    return int(match.group(1)), int(match.group(2))


def setup_ops_extension():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    nvcc_args = [
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-gencode=arch=compute_100,code=sm_100",
        "-DNDEBUG",
    ]
    cuda_major, cuda_minor = get_cuda_version()
    if cuda_major < 12:
        raise ValueError(f"CUDA version must be >= 12. Detected version: {cuda_major}.{cuda_minor}")
    if cuda_major == 12 and cuda_minor < 8:
        nvcc_args = [arg for arg in nvcc_args if "compute_100" not in arg]

    change_pwd()
    ext_module = CUDAExtension(
        sources=["./flashmask_cuda_utils_op.cu"],
        include_dirs=[os.getcwd()],
        extra_compile_args={
            "cxx": ["-O3", "-w", "-Wno-abi", "-fPIC", "-std=c++17"],
            "nvcc": nvcc_args,
        },
    )

    setup(
        name="flashmask_cuda_utils",
        ext_modules=[ext_module],
        version="0.0.1",
    )


setup_ops_extension()
