'''setup.py for archngv-building'''
import importlib
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


spec = importlib.util.spec_from_file_location("ngv_ctools.version", "ngv_ctools/version.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION


compiler_args = ["-DNDEBUG", "-O3"]
macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]


extensions = [
    Extension("ngv_ctools.endfeet_reconstruction.fmm_growing",
              ["ngv_ctools/endfeet_reconstruction/fmm_growing.pyx"],
              define_macros=macros, extra_compile_args=compiler_args),
    Extension("ngv_ctools.endfeet_reconstruction.local_solvers",
              ["ngv_ctools/endfeet_reconstruction/local_solvers.pyx"],
              define_macros=macros, extra_compile_args=compiler_args),
    Extension("ngv_ctools.endfeet_reconstruction.priority_heap",
              ["ngv_ctools/endfeet_reconstruction/priority_heap.pyx"],
              define_macros=macros, extra_compile_args=compiler_args)
]


class LazyImportBuildExtCmd(build_ext):
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        super().run()

    def finalize_options(self):
        from Cython.Build import cythonize
        self.distribution.ext_modules = cythonize(self.distribution.ext_modules)
        super().finalize_options()


setup(
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    name='ngv-ctools',
    version=VERSION,
    description='NGV Architectur_bui Cython Building Modules',
    author='Eleftherios Zisis',
    author_email='eleftherios.zisis@epfl.ch',
    url='https://bbpgitlab.epfl.ch/molsys/ngv-ctools',
    packages=find_packages(),
    cmdclass={'build_ext': LazyImportBuildExtCmd},
    ext_modules=extensions,
    include_package_data=True,
    setup_requires=[
        'numpy>=1.13',
    ],
    install_requires=[
        'numpy>=1.13'
    ],
)
