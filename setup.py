'''setup.py for archngv-building'''
from setuptools import setup, Extension, find_packages
import numpy

spec = importlib.util.spec_from_file_location("ngv_ctools.version", "ngv_ctools/version.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION


compiler_args = ["-DNDEBUG", "-O3"]
macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]


extensions = [
    Extension("*", ["ngv_ctools/endfeet_reconstruction/*.pyx"], define_macros=macros, extra_compile_args=compiler_args, include_dirs=[numpy.get_include()])
]

setup(
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
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
