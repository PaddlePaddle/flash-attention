# Adapted from https://github.com/NVIDIA/apex/blob/master/setup.py
import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform
import shutil
from setuptools import Command, Extension, setup
from setuptools.command.develop import develop as DevelopCommandBase
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install as InstallCommandBase
from setuptools.command.install_lib import install_lib
from setuptools.dist import Distribution
from setuptools import setup, find_packages
import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import paddle
from paddle.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from paddle.utils.cpp_extension.extension_utils import find_cuda_home
import subprocess

version_detail = sys.version_info
python_version = platform.python_version()
version = version_detail[0] + version_detail[1] / 10
env_version = os.getenv("PY_VERSION")

if version < 3.7:
    raise RuntimeError(
        f"Paddle only supports Python version >= 3.7 now,"
        f"you are using Python {python_version}"
    )
elif env_version is None:
    print(f"export PY_VERSION = { python_version }")
    os.environ["PY_VERSION"] = python_version

elif env_version != version:
    warnings.warn(
        f"You set PY_VERSION={env_version}, but"
        f"your current python environment is {version}"
        f"we will use your current python version to execute"
    )
    os.environ["PY_VERSION"] = python_version

paddle_include_path = paddle.sysconfig.get_include()
paddle_lib_path = paddle.sysconfig.get_lib()

print("Paddle Include Path:", paddle_include_path)
print("Paddle Lib Path:", paddle_lib_path)

# preparing parameters for setup()
paddle_version = paddle.version.full_version
cuda_version= paddle.version.cuda_version


with open("../../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = find_cuda_home()
PACKAGE_NAME = "paddle_flash_attn"

BASE_WHEEL_URL = "https://github.com/PaddlePaddle/flash-attention/releases/download/{tag_name}/{wheel_name}"

FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"
# For CI, we want the option to not add "--threads 4" to nvcc, since the runner can OOM
FORCE_SINGLE_THREAD = os.getenv("FLASH_ATTENTION_FORCE_SINGLE_THREAD", "FALSE") == "TRUE"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith('linux'):
        return 'linux_x86_64'
    elif sys.platform == 'darwin':
        mac_version = '.'.join(platform.mac_ver()[0].split('.')[:2])
        return f'macosx_{mac_version}_x86_64'
    elif sys.platform == 'win32':
        return 'win_amd64'
    else:
        raise ValueError('Unsupported platform: {}'.format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_cuda_paddle_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    paddle_binary_version = parse(paddle.version.cuda())

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_version != paddle_binary_version):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pypaddle binaries.  "
            "Pypaddle binaries were compiled with Cuda {}.\n".format(paddle.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )


def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pypaddle/pypaddle, "
        "only images whose names contain 'devel' will provide nvcc."
    )

def _is_cuda_available():
    """
    Check whether CUDA is available.
    """
    try:
        assert len(paddle.static.cuda_places()) > 0
        return True
    except Exception as e:
        logging.warning(
            "You are using GPU version PaddlePaddle, but there is no GPU "
            "detected on your machine. Maybe CUDA devices is not set properly."
            f"\n Original Error is {e}"
        )
        return False

cmdclass = {}

def get_package_version():
    with open(Path(this_dir) / "../flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)

def get_data_files():
    data_files = []
    
    # Assuming 'libflashattn.so' is located in the same directory as setup.py
    source_lib_path = 'libflashattn.so'

    # Specify the destination directory within the package
    destination_lib_path = os.path.join(PACKAGE_NAME, 'build/libflashattn.so')

    data_files.append((paddle_lib_path, [source_lib_path]))
    return data_files


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all flash attention installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """
    def run(self):
        print("88888888888888888888888888888")
       # if FORCE_BUILD:
       #     return super().run()
        self.run_command('build_ext')
        super().run()
        # Determine the version numbers that will be used to determine the correct wheel
        # We're using the CUDA version used to build paddle, not the one currently installed
        # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
        paddle_cuda_version = "234" #parse(paddle.version.cuda)
        paddle_version_raw = parse(paddle.__version__)
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform_name = get_platform()
        flash_version = get_package_version()
        cxx11_abi ="" # str(paddle._C.-D_GLIBCXX_USE_CXX11_ABI).upper()

        # Determine wheel URL based on CUDA version, paddle version, python version and OS
        wheel_filename = f'{PACKAGE_NAME}-{flash_version}-cu{cuda_version}-paddle{paddle_version}-{python_version}-{python_version}-{platform_name}.whl'
        impl_tag, abi_tag, plat_tag = self.get_tag()
        original_wheel_name = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

        new_wheel_name ='asdfsdf.whl' # wheel_filename
        #shutil.move(
        #    f"{self.dist_dir}/{original_wheel_name}.whl",
        #    f"{self.dist_dir}/{new_wheel_name}"
        #) 


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(),
    data_files=get_data_files(),
    package_data={PACKAGE_NAME: ['build/libflashattn.so']},
    author_email="Paddle-better@baidu.com",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PaddlePaddle/flash-attention",
    classifiers=[
        "Programming Language :: Python :: 37",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    cmdclass={
    "bdist_wheel": CachedWheelsCommand,},
    python_requires=">=3.7",
    install_requires=[
        "paddle",
        "einops",
        "packaging",
        "ninja",
    ],
)
