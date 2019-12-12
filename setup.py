'''setup.py for archngv-building'''
import imp
from setuptools import setup, Extension, find_packages

try:
    from Cython.Build import cythonize
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False

VERSION = imp.load_source("archngv_building.version", "archngv_building/version.py").VERSION


if _USE_CYTHON:
    extensions = cythonize([Extension("*",
                                      ["archngv_building/endfeet_reconstruction/*.pyx", ],),
                            ])
else:
    from glob import glob
    sources = glob("archngv_building/endfeet_reconstruction/*.c")
    assert sources, 'Must have .c files in archngv_building/endfeet_reconstruction/'
    extensions = [Extension(source.strip('.c'), [source])
                  for source in sources]



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
    ext_modules=extensions,
    include_package_data=True,
    setup_requires=[
        'numpy>=1.13',
    ],
    install_requires=[
        'numpy>=1.13',
    ],
)
