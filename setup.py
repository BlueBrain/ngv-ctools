'''setup.py for archngv-building'''
import imp
from setuptools import setup, Extension, find_packages
import numpy
try:
    from Cython.Build import cythonize
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False

VERSION = imp.load_source("ngv_ctools.version", "ngv_ctools/version.py").VERSION


if _USE_CYTHON:
    extensions = cythonize([Extension("*",
                                      ["ngv_ctools/endfeet_reconstruction/*.pyx", ], include_dirs=[numpy.get_include()]),
                            ])
else:
    from glob import glob
    sources = glob("ngv_ctools/endfeet_reconstruction/*.c")
    assert sources, 'Must have .c files in ngv_ctools/endfeet_reconstruction/'
    extensions = [Extension(source.strip('.c'), [source])
                  for source in sources]



setup(
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
    name='ngv_ctools',
    version=VERSION,
    description='NGV Architectur_bui Cython Building Modules',
    author='Eleftherios Zisis',
    author_email='eleftherios.zisis@epfl.ch',
    packages=find_packages(),
    ext_modules=extensions,
    include_package_data=True,
    setup_requires=[
        'numpy>=1.13',
    ],
    install_requires=[
        'numpy>=1.13'
    ],
)
