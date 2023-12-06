from setuptools import setup, find_packages
from setuptools import setup, find_namespace_packages

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"": ["*.so"]},
    exclude_package_data={"flash_attn_with_bias_and_mask": ["*"]},
    include_package_data=True,
    #packages=find_namespace_packages(where="src"),
    #package_dir={"": "src"},
    #package_data={
    #    "": ["*.so"],
    #}
)
