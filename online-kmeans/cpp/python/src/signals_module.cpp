#include <pybind11/pybind11.h>

#include "sequential_submodule.hpp"
#include "parallel_submodule.hpp"

namespace signals {

} // namespace signals

namespace py = pybind11;


static void submodules(pybind11::module& m)
{
    auto seq_subm = m.def_submodule("seq");
    sequential_submodule(seq_subm);

    auto par_subm = m.def_submodule("par");
    parallel_submodule(par_subm);
}

PYBIND11_MODULE(pysignals, m) {
    m.doc() = "Python signals core module";
    submodules(m);
}