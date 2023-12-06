from setuptools import setup

package_name = '' #flash-attention-paddle-gpu'
setup(
    name=package_name,
    version='1.0.0',
    description='Flash attention in PaddlePaddle',
    packages=[package_name],
    include_package_data=True,
    package_data={package_name: ['csrc/build/libflashattn.so']},
)

