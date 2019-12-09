'''setup.py for archngv-building'''
import imp
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

VERSION = imp.load_source("archngv_building.version", "archngv_building/version.py").VERSION

cython_extensions = [Extension("*", ["archngv_building/endfeet_reconstruction/*.pyx", ],),
                     ]

setup(
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
    name='archngv_building',
    version=VERSION,
    description='NGV Architecture Cython Building Modules',
    author='Eleftherios Zisis',
    author_email='eleftherios.zisis@epfl.ch',
    packages=find_packages(),
    ext_modules=cythonize(cython_extensions),
    include_package_data=True,
    install_requires=[
        'numpy>=1.13',
    ],
)
