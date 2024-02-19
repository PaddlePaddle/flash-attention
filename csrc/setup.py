# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import os
import re
import subprocess
import sys
from pathlib import Path

from env_dict import env_dict
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


with open("../../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


cur_dir = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "paddle-flash-attn"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith('linux'):
        return 'linux_x86_64'
    elif sys.platform == 'win32':
        return 'win_amd64'
    else:
        raise ValueError(f'Unsupported platform: {sys.platform}')


def get_cuda_version():
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if line.startswith('Cuda compilation tools'):
                    cuda_version = (
                        line.split('release')[1].strip().split(',')[0]
                    )
                    return cuda_version
        else:
            print("Error:", result.stderr)

    except Exception as e:
        print("Error:", str(e))

    return None


def get_package_version():
    with open(Path(cur_dir) / "../flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(
            r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE
        )
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


def get_package_data():
    binary_dir = env_dict.get("CMAKE_BINARY_DIR")
    lib = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        binary_dir + '/paddle_flash_attn/lib/libflashattn_advanced.so',
    )
    package_data = {'paddle_flash_attn': [lib]}
    return package_data


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
        cuda_version = get_cuda_version()
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform_name = get_platform()
        flash_version = get_package_version()
        wheel_name = 'paddle_flash_attn'

        # Determine wheel URL based on CUDA version, python version and OS
        impl_tag, abi_tag, plat_tag = self.get_tag()
        archive_basename = (
            f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"
        )
        wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
        print("Raw wheel path", wheel_path)
        wheel_filename = f'{wheel_name}-{flash_version}+cu{cuda_version}-{impl_tag}-{abi_tag}-{platform_name}.whl'
        os.rename(wheel_path, os.path.join(self.dist_dir, wheel_filename))


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=['paddle_flash_attn'],
    package_data=get_package_data(),
    author_email="Paddle-better@baidu.com",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PaddlePaddle/flash-attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    cmdclass={
        'bdist_wheel': CustomWheelsCommand,
    },
    python_requires=">=3.7",
)
