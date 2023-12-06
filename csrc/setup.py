# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# Adapted from https://github.com/NVIDIA/apex/blob/master/setup.py
import ast
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

from packaging.version import parse
from setuptools import find_packages, setup
from setuptools.command.install import install as _install
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import paddle
from paddle.utils.cpp_extension.extension_utils import find_cuda_home

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
cuda_version = paddle.version.cuda_version


with open("../../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = find_cuda_home()
PACKAGE_NAME = "paddle_flash_attn"


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
        raise ValueError(f'Unsupported platform: {sys.platform}')


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


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


check = _is_cuda_available()
cmdclass = {}


def get_package_version():
    with open(Path(this_dir) / "../flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(
            r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE
        )
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


def get_data_files():
    data_files = []
    source_lib_path = 'libflashattn.so'
    data_files.append((".", [source_lib_path]))
    return data_files


class CustomWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all flash attention installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        self.run_command('build_ext')
        super().run()
        # Determine the version numbers that will be used to determine the correct wheel
        # We're using the CUDA version used to build paddle, not the one currently installed
        # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
        paddle_cuda_version = "234"  # parse(paddle.version.cuda)
        paddle_version_raw = parse(paddle.__version__)
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform_name = get_platform()
        flash_version = get_package_version()
        cxx11_abi = ""  # str(paddle._C.-D_GLIBCXX_USE_CXX11_ABI).upper()

        # Determine wheel URL based on CUDA version, paddle version, python version and OS
        wheel_filename = f'{PACKAGE_NAME}-{flash_version}-cu{cuda_version}-paddle{paddle_version}-{python_version}-{python_version}-{platform_name}.whl'
        impl_tag, abi_tag, plat_tag = self.get_tag()
        original_wheel_name = (
            f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"
        )

        # new_wheel_name = wheel_filename
        new_wheel_name = (
            f"{self.wheel_dist_name}-{python_version}-{abi_tag}-{plat_tag}"
        )
        shutil.move(
            f"{self.dist_dir}/{original_wheel_name}.whl",
            f"{self.dist_dir}/{new_wheel_name}.whl",
        )


class CustomInstallCommand(_install):
    def run(self):
        _install.run(self)
        install_path = self.install_lib
        # src
        source_lib_path = os.path.abspath('libflashattn.so')

        destination_lib_path = os.path.join(paddle_lib_path, 'libflashattn.so')

        # shutil.move(f"{source_lib_path}", f"{destination_lib_path}")
        # os.symlink(source_lib_path, destination_lib_path)


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
        'bdist_wheel': CustomWheelsCommand,
        'install': CustomInstallCommand,
    },
    python_requires=">=3.7",
    install_requires=[
        "paddle",
        "einops",
        "packaging",
        "ninja",
    ],
)
