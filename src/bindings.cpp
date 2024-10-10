#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fast_marching_method.hpp" // BindFastMarchingMethodModule

PYBIND11_MODULE(_ngv_ctools, m) { ngv::BindFastMarchingMethodModule(m); }
