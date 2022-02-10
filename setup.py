"""setup.py for archngv-building"""
import importlib.util
from setuptools import find_packages, setup


try:
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    # the purpose of this hack is so that publish-package ci job
    # can execute python setup.py --name and --version without
    # stumbling on the pybind11 import
    from setuptools import Extension as Pybind11Extension


spec = importlib.util.spec_from_file_location(
    "ngv_ctools.version", "ngv_ctools/version.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION


ext_modules = [
    Pybind11Extension(
        "_ngv_ctools",
        ["src/bindings.cpp"],
        include_dirs=["include/"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    )
]


setup(
    name="ngv-ctools",
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    version=VERSION,
    description="NGV Architecture c++ modules",
    author="Eleftherios Zisis",
    author_email="eleftherios.zisis@epfl.ch",
    url="https://bbpgitlab.epfl.ch/molsys/ngv-ctools",
    packages=find_packages(),
    ext_modules=ext_modules,
    include_package_data=True,
)
