"""setup.py for archngv-building"""
import importlib
from glob import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

spec = importlib.util.spec_from_file_location(
    "ngv_ctools.version", "ngv_ctools/version.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION


ext_modules = [
    Pybind11Extension(
        "_ngv_ctools",
        sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
        include_dirs=["include/"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    ),
]


setup(
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    name="ngv-ctools",
    version=VERSION,
    description="NGV Architecture c++ modules",
    author="Eleftherios Zisis",
    author_email="eleftherios.zisis@epfl.ch",
    url="https://bbpgitlab.epfl.ch/molsys/ngv-ctools",
    packages=find_packages(),
    ext_modules=ext_modules,
    include_package_data=True,
)
