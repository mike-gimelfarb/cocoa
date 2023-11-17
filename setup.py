from setuptools import setup
import os

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys

__version__ = "0.0.1"

def list_source_files(root_folder):
    cppfiles = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.cpp'):
                cppfiles.append(os.path.join(root, file))
    return cppfiles


ext_modules = [
    Pybind11Extension(
        "cocoaopt",
        list_source_files('py') + list_source_files('src') + list_source_files(''),
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    name="cocoaopt",
    version=__version__,
    author="Mike Gimelfarb",
    description="Collection of Continuous Optimization Algorithms",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
