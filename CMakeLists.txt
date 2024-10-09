# Set the minimum CMake version and policies for highest tested version
cmake_minimum_required(VERSION 3.15...3.27)

project(
  # name of the project
  ngv_ctools

  DESCRIPTION "NGV C++ extensions"

  # ensure there is a working c++ compiler
  LANGUAGES CXX
)

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_ngv_ctools
    ${CMAKE_CURRENT_SOURCE_DIR}/src/types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/priority_heap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fast_marching_method.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp
)

# Install directive for scikit-build-core
install(TARGETS _ngv_ctools LIBRARY DESTINATION ngv_ctools)
