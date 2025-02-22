#include "sequential_submodule.hpp"

#include "core.hpp"

#include <iostream>

namespace signals {

} // namespace signals


using namespace signals;


void sequential_submodule(pybind11::module& subm)
{
    subm.doc() = "Python signals submodule for sequential algorithms";

    subm.def(
        "k_means",
        [](py::array_t<std::uint8_t> img_arr,
           std::uint64_t k, std::float_t stab_error, int max_iterations) {

            py::buffer_info img_buf = img_arr.request();

            if (img_buf.ndim != 3)
                throw std::runtime_error("Number of dimensions must be 3");
            
            auto result = py::array_t<std::uint8_t>(
                {img_buf.shape[0], img_buf.shape[1], img_buf.shape[2]});
            result[py::make_tuple(py::ellipsis())] = 0;
            py::buffer_info result_buf = result.request();

            k_means(
                static_cast<std::uint8_t*>(result_buf.ptr),
                static_cast<std::uint8_t*>(img_buf.ptr),
                img_buf.shape[0], img_buf.shape[1],
                k, stab_error, max_iterations
            );

            return result;
        });
}