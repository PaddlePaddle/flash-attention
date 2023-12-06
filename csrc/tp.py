import paddle
from setuptools import setup, find_packages
import sys
import os
import paddle
paddle_path = paddle.sysconfig.get_lib
print(paddle_path)
python_version = sys.version
print("Installing your_package...")

# Get the CUDA version from PaddlePaddle
cuda_version = paddle.version.cuda()
fa_version = f"1.0.0.post{cuda_version}"
package_name = 'flash_attention_paddle_gpu'

def get_data_files():
    data_files = []
    
    # Assuming 'libflashattn.so' is located in the same directory as setup.py
    source_lib_path = 'libflashattn.so'

    # Specify the destination directory within the package
    destination_lib_path = os.path.join(package_name, 'libflashattn.so')

    data_files.append((os.path.join(package_name, 'libflashattn.so'), [source_lib_path]))
    print(destination_lib_path, "asdf ****************")
    print(data_files)
    return data_files

setup(
    name=package_name,
    version=fa_version,
    data_files=get_data_files(),
    description='Flash attention in paddlepaddle',
    packages=find_packages(),
    package_data={package_name: ['build/libflashattn.so']},
)
#
#import paddle
#import os
#from setuptools import setup
#import sys
#
#python_version = sys.version
#print("Installing your_package...")
#
## Get the CUDA version from PaddlePaddle
#cuda_version = paddle.version.cuda()
#fa_version = f"1.0.0.post{cuda_version}"
#package_name = 'flash_attention_paddle_gpu'  # Adjusted package name
#
#def get_data_files():
#    data_files = []
#    
#    # Assuming 'libflashattn.so' is located in the same directory as setup.py
#    source_lib_path = os.path.abspath('libflashattn.so')
#
#    # Specify the destination directory within the package
#    destination_lib_path = os.path.join(package_name, 'libflashattn.so')
#
#    data_files.append((os.path.join(package_name, 'libflashattn.so'), [source_lib_path]))
#    print(destination_lib_path, "asdf ****************")
#    print(data_files)
#    return data_files
#
## Create an empty __init__.py file in the package directory
#init_file_path = os.path.join(package_name, '__init__.py')
#with open(init_file_path, 'w') as f:
#    pass
#
#setup(
#    name=package_name,
#    version=fa_version,
#    description='Flash attention in paddlepaddle',
#    packages=[package_name],
#    package_data={package_name: ['libflashattn.so']},
#)
