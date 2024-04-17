"""setup.py for archngv-building"""
from setuptools import find_packages, setup


try:
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    # the purpose of this hack is so that publish-package ci job
    # can execute python setup.py --name and --version without
    # stumbling on the pybind11 import
    from setuptools import Extension as Pybind11Extension


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
    python_requires=">=3.8",
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "local_scheme": "no-local-version",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    description="NGV Architecture c++ modules",
    author="Eleftherios Zisis",
    author_email="eleftherios.zisis@epfl.ch",
    url="https://bbpgitlab.epfl.ch/molsys/ngv-ctools",
    packages=find_packages(),
    ext_modules=ext_modules,
    include_package_data=True,
)
