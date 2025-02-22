#ifndef SIGNALS_PYTHON_SEQUENTIAL_SUBMODULE_HPP
#define SIGNALS_PYTHON_SEQUENTIAL_SUBMODULE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace pybind11::literals;


namespace signals {

} // namespace signals


void sequential_submodule(pybind11::module& module);

#endif // SIGNALS_PYTHON_SEQUENTIAL_SUBMODULE_HPP