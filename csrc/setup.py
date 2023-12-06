# Adapted from https://github.com/NVIDIA/apex/blob/master/setup.py
import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import Command, Extension, setup
from setuptools.command.develop import develop as DevelopCommandBase
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install as InstallCommandBase
from setuptools.command.install_lib import install_lib
from setuptools.dist import Distribution
from setuptools import setup, find_packages
import subprocess
python_version = platform.python_version()
version_detail = sys.version_info
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


global env_dict  # noqa: F811
env_dict={
    'PADDLE_SOURCE_DIR':'@PADDLE_SOURCE_DIR@',
    'PADDLE_VERSION':'@PADDLE_VERSION@',
    'PADDLE_BINARY_DIR':'@PADDLE_BINARY_DIR@',
    'TAG_VERSION_REGEX':'@TAG_VERSION_REGEX@',
    'WITH_GPU':'@WITH_GPU@',
    'CUDNN_MAJOR_VERSION':'@CUDNN_MAJOR_VERSION@',
    'CUDNN_MINOR_VERSION':'@CUDNN_MINOR_VERSION@',
    'CUDNN_PATCHLEVEL_VERSION':'@CUDNN_PATCHLEVEL_VERSION@',
    'CUDA_VERSION':'@CUDA_VERSION@',
    'WITH_PSLI':'@WITH_PSLI@',
    'FLUID_CORE_NAME':'@FLUID_CORE_NAME@',
    'PHI_LIB':'@PHI_LIB@',
    'PHI_NAME':'@PHI_NAME@',
    'WITH_SHARED_PHI':'@WITH_SHARED_PHI@',
    'IR_LIB':'@IR_LIB@',
    'IR_NAME':'@IR_NAME@',
    'WITH_SHARED_IR':'@WITH_SHARED_IR@',
    'COMMON_LIB':'@COMMON_LIB@',
    'COMMON_NAME':'@COMMON_NAME@',
    'WARPCTC_LIBRARIES':'@WARPCTC_LIBRARIES@',
    'WARPRNNT_LIBRARIES':'@WARPRNNT_LIBRARIES@',
    'FLASHATTN_LIBRARIES':'@FLASHATTN_LIBRARIES@',
    'LAPACK_LIB':'@LAPACK_LIB@',
    'GFORTRAN_LIB':'@GFORTRAN_LIB@',
    'GNU_RT_LIB_1':'@GNU_RT_LIB_1@',
    'WITH_CUDNN_DSO':'@WITH_CUDNN_DSO@',
    'CUDNN_LIBRARY':'@CUDNN_LIBRARY@',
    'GNU_RT_LIB_2':'@GNU_RT_LIB_2@',
    'WITH_MKL':'@WITH_MKL@',
    'MKLML_SHARED_LIB':'@MKLML_SHARED_LIB@',
    'MKLML_SHARED_IOMP_LIB':'@MKLML_SHARED_IOMP_LIB@',
    'OPENBLAS_SHARED_LIB':'@OPENBLAS_SHARED_LIB@',
    'OPENBLAS_LIB':'@OPENBLAS_LIB@',
    'BLAS_LIB':'@BLAS_LIB@',
    'WITH_LITE':'@WITH_LITE@',
    'LITE_SHARED_LIB':'@LITE_SHARED_LIB@',
    'LITE_WITH_NNADAPTER':'@LITE_WITH_NNADAPTER@',
    'LITE_NNADAPTER_LIB':'@LITE_NNADAPTER_LIB@',
    'NNADAPTER_WITH_HUAWEI_ASCEND_NPU':'@NNADAPTER_WITH_HUAWEI_ASCEND_NPU@',
    'LITE_NNADAPTER_NPU_LIB':'@LITE_NNADAPTER_NPU_LIB@',
    'WITH_CINN':'@WITH_CINN@',
    'CINN_LIB_LOCATION':'@CINN_LIB_LOCATION@',
    'CINN_LIB_NAME':'@CINN_LIB_NAME@',
    'CINN_INCLUDE_DIR':'@CINN_INCLUDE_DIR@',
    'CMAKE_BUILD_TYPE':'@CMAKE_BUILD_TYPE@',
    'PSLIB_LIB':'@PSLIB_LIB@',
    'JVM_LIB':'@JVM_LIB@',
    'PSLIB_VERSION_PY':'@PSLIB_VERSION_PY@',
    'WITH_MKLDNN':'@WITH_MKLDNN@',
    'MKLDNN_SHARED_LIB':'@MKLDNN_SHARED_LIB@',
    'MKLDNN_INSTALL_DIR':'@MKLDNN_INSTALL_DIR@',
    'WITH_ONNXRUNTIME':'@WITH_ONNXRUNTIME@',
    'ONNXRUNTIME_SHARED_LIB':'@ONNXRUNTIME_SHARED_LIB@',
    'PADDLE2ONNX_LIB':'@PADDLE2ONNX_LIB@',
    'PADDLE2ONNX_LIB_NAME':'@PADDLE2ONNX_LIB_NAME@',
    'ONNXRUNTIME_LIB_NAME':'@ONNXRUNTIME_LIB_NAME@',
    'WITH_XPU':'@WITH_XPU@',
    'XPU_API_LIB':'@XPU_API_LIB@',
    'XPU_API_LIB_NAME':'@XPU_API_LIB_NAME@',
    'XPU_RT_LIB':'@XPU_RT_LIB@',
    'XPU_RT_LIB_NAME':'@XPU_RT_LIB_NAME@',
    'WITH_XPU_BKCL':'@WITH_XPU_BKCL@',
    'XPU_BKCL_LIB':'@XPU_BKCL_LIB@',
    'XPU_BKCL_LIB_NAME':'@XPU_BKCL_LIB_NAME@',
    'WITH_XPU_XFT':'@WITH_XPU_XFT@',
    'XPU_XFT_LIB':'@XPU_XFT_LIB@',
    'XPU_XFT_LIB_NAME':'@XPU_XFT_LIB_NAME@',
    'THIRD_PARTY_PATH':'@THIRD_PARTY_PATH@',
    'SETUP_LOG_FILE':'@SETUP_LOG_FILE@',
    'WITH_STRIP':'@WITH_STRIP@',
    'PACKAGE_NAME':'@PACKAGE_NAME@',
    'PADDLE_VERSION':'@PADDLE_VERSION@',
    'APPLE':'@APPLE@',
    'externalError_INCLUDE_DIR':'@externalError_INCLUDE_DIR@',
    'WITH_ROCM':'@WITH_ROCM@',
    'ORIGIN':'@ORIGIN@',
    'WIN32':'@WIN32@',
    'JIT_RELEASE_WHL':'@JIT_RELEASE_WHL@',
    'WITH_PSLIB':'@WITH_PSLIB@',
    'PYBIND_INCLUDE_DIR':'@PYBIND_INCLUDE_DIR@',
    'WITH_PYTHON':'@WITH_PYTHON@',
    'WITH_CINN':'@WITH_CINN@',
    'CINN_SOURCE_DIR':'@CINN_SOURCE_DIR@',
    'WITH_CPP_DIST':'@WITH_CPP_DIST@',
    'PADDLE_INSTALL_DIR':'@PADDLE_INSTALL_DIR@',
    'PADDLE_LIB_TEST_DIR':'@PADDLE_LIB_TEST_DIR@'
}

global paddle_binary_dir, paddle_source_dir

paddle_binary_dir = env_dict.get("PADDLE_BINARY_DIR")
paddle_source_dir = env_dict.get("PADDLE_SOURCE_DIR")

# preparing parameters for setup()
paddle_version = env_dict.get("PADDLE_VERSION")
package_name = env_dict.get("PACKAGE_NAME")

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import paddle
from paddle.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from paddle.utils.cpp_extension.extension_utils import find_cuda_home

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = find_cuda_home()
PACKAGE_NAME = "flash_attn"

BASE_WHEEL_URL = "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
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


def append_nvcc_threads(nvcc_extra_args):
    if not FORCE_SINGLE_THREAD:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args

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

if paddle.is_compiled_with_cuda() and _is_cuda_available():
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pypaddle/pypaddle/pull/23408 attempt to query paddle.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print(
        "\nWarning: Torch did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, FlashAttention will cross-compile for Ampere (compute capability 8.0, 8.6, "
        "8.9), and, if the CUDA version is >= 11.8, Hopper (compute capability 9.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export PADDLE_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )
    if os.environ.get("PADDLE_CUDA_ARCH_LIST", None) is None and CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version >= Version("11.8"):
            os.environ["PADDLE_CUDA_ARCH_LIST"] = "8.0;8.6;9.0"
        elif bare_metal_version >= Version("11.4"):
            os.environ["PADDLE_CUDA_ARCH_LIST"] = "8.0;8.6"
        else:
            os.environ["PADDLE_CUDA_ARCH_LIST"] = "8.0;8.6"

cmdclass = {}
ext_modules = []

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

    data_files.append((os.path.join(PACKAGE_NAME, 'libflashattn.so'), [source_lib_path]))
    return data_files


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all flash attention installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """
    def run(self):
        if FORCE_BUILD:
            return super().run()

        # Determine the version numbers that will be used to determine the correct wheel
        # We're using the CUDA version used to build paddle, not the one currently installed
        # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
        paddle_cuda_version = parse(paddle.version.cuda)
        paddle_version_raw = parse(paddle.__version__)
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform_name = get_platform()
        flash_version = get_package_version()
        # cuda_version = f"{cuda_version_raw.major}{cuda_version_raw.minor}"
        cuda_version = f"{paddle_cuda_version.major}{paddle_cuda_version.minor}"
        paddle_version = f"{paddle_version_raw.major}.{paddle_version_raw.minor}"
        cxx11_abi = str(paddle._C._GLIBCXX_USE_CXX11_ABI).upper()

        # Determine wheel URL based on CUDA version, paddle version, python version and OS
        wheel_filename = f'{PACKAGE_NAME}-{flash_version}+cu{cuda_version}paddle{paddle_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl'
        wheel_url = BASE_WHEEL_URL.format(
            tag_name=f"v{flash_version}",
            wheel_name=wheel_filename
        )
        print("Guessing wheel URL: ", wheel_url)

        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            os.rename(wheel_filename, wheel_path)
        except urllib.error.HTTPError:
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()

class InstallHeaders(Command):
    """Override how headers are copied."""

    description = 'install C/C++ header files'

    user_options = [
        ('install-dir=', 'd', 'directory to install header files to'),
        ('force', 'f', 'force installation (overwrite existing files)'),
    ]

    boolean_options = ['force']

    def initialize_options(self):
        self.install_dir = None
        self.force = 0
        self.outfiles = []

    def finalize_options(self):
        self.set_undefined_options(
            'install', ('install_headers', 'install_dir'), ('force', 'force')
        )

    def run(self):
        hdrs = self.distribution.headers
        if not hdrs:
            return
        self.mkpath(self.install_dir)
        for header in hdrs:
            install_dir = get_header_install_dir(header)
            install_dir = os.path.join(
                self.install_dir, os.path.dirname(install_dir)
            )
            if not os.path.exists(install_dir):
                self.mkpath(install_dir)
            (out, _) = self.copy_file(header, install_dir)
            self.outfiles.append(out)
            # (out, _) = self.mkdir_and_copy_file(header)
            # self.outfiles.append(out)

    def get_inputs(self):
        return self.distribution.headers or []

    def get_outputs(self):
        return self.outfiles


class InstallCommand(InstallCommandBase):
    def finalize_options(self):
        ret = InstallCommandBase.finalize_options(self)
        self.install_lib = self.install_platlib

        self.install_headers = os.path.join(
            self.install_platlib, 'paddle', 'include'
        )
        return ret


class DevelopCommand(DevelopCommandBase):
    def run(self):
        # copy proto and .so to python_source_dir
        fluid_proto_binary_path = (
            paddle_binary_dir + '/python/paddle/base/proto/'
        )
        fluid_proto_source_path = (
            paddle_source_dir + '/python/paddle/base/proto/'
        )
        distributed_proto_binary_path = (
            paddle_binary_dir + '/python/paddle/distributed/fleet/proto/'
        )
        distributed_proto_source_path = (
            paddle_source_dir + '/python/paddle/distributed/fleet/proto/'
        )
        os.system(f"rm -rf {fluid_proto_source_path}")
        shutil.copytree(fluid_proto_binary_path, fluid_proto_source_path)
        os.system(f"rm -rf {distributed_proto_source_path}")
        shutil.copytree(
            distributed_proto_binary_path, distributed_proto_source_path
        )
        shutil.copy(
            paddle_binary_dir + '/python/paddle/base/libpaddle.so',
            paddle_source_dir + '/python/paddle/base/',
        )
        dynamic_library_binary_path = paddle_binary_dir + '/python/paddle/libs/'
        dynamic_library_source_path = paddle_source_dir + '/python/paddle/libs/'
        for lib_so in os.listdir(dynamic_library_binary_path):
            shutil.copy(
                dynamic_library_binary_path + lib_so,
                dynamic_library_source_path,
            )
        # write version.py and cuda_env_config_py to python_source_dir
        write_version_py(
            filename=f'{paddle_source_dir}/python/paddle/version/__init__.py'
        )
        write_cuda_env_config_py(
            filename=f'{paddle_source_dir}/python/paddle/cuda_env.py'
        )
        write_parameter_server_version_py(
            filename='{}/python/paddle/incubate/distributed/fleet/parameter_server/version.py'.format(
                paddle_source_dir
            )
        )
        DevelopCommandBase.run(self)


class EggInfo(egg_info):
    """Copy license file into `.dist-info` folder."""

    def run(self):
        # don't duplicate license into `.dist-info` when building a distribution
        if not self.distribution.have_run.get('install', True):
            self.mkpath(self.egg_info)
            #self.copy_file(
            #    env_dict.get("PADDLE_SOURCE_DIR") + "/LICENSE", self.egg_info
            #)

        egg_info.run(self)


# class Installlib is rewritten to add header files to .egg/paddle
class InstallLib(install_lib):
    def run(self):
        self.build()
        outfiles = self.install()
        hrds = self.distribution.headers
        if not hrds:
            return
        for header in hrds:
            install_dir = get_header_install_dir(header)
            install_dir = os.path.join(
                self.install_dir, 'paddle/include', os.path.dirname(install_dir)
            )
            if not os.path.exists(install_dir):
                self.mkpath(install_dir)
            self.copy_file(header, install_dir)
        if outfiles is not None:
            # always compile, in case we have any extension stubs to deal with
            self.byte_compile(outfiles)



setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        #exclude=("build")
    #, "csrc", "include", "tests", "dist", "docs", "benchmarks", "flash_attn.egg-info",)
    ),
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
    ext_modules=ext_modules,
    cmdclass={
        'install_headers': InstallHeaders,
        'install': InstallCommand,
        'egg_info': EggInfo,
        'install_lib': InstallLib,
        'develop': DevelopCommand,
    },
    #cmdclass={
    #    "bdist_wheel": CachedWheelsCommand,
    #    "build_ext": BuildExtension
    #} if ext_modules else {
    #    "bdist_wheel": CachedWheelsCommand,
    #},
    python_requires=">=3.7",
    install_requires=[
        "paddle",
        "einops",
        "packaging",
        "ninja",
    ],
)
