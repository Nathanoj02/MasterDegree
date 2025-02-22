#include "parallel_submodule.hpp"

#include "core_par.hpp"

namespace signals {

} // namespace signals


using namespace signals;


void parallel_submodule(pybind11::module& subm)
{
    subm.doc() = "Python signals submodule for parallel algorithms";

    py::class_<cuda::DeviceInfo>(subm, "DeviceInfo");
    py::class_<cuda::KmeansInfo>(subm, "KmeansInfo");

    subm.def(
        "init_k_means", 
        [](std::size_t img_height, std::size_t img_width, std::uint64_t k) {
            auto dev = cuda::init_k_means(img_height, img_width, k);
            return dev;
        });

    subm.def(
        "k_means",
        [](py::array_t<std::uint8_t> img_arr,
           std::uint64_t k, std::float_t stab_error, int max_iterations,
           cuda::KmeansInfo& dev_info, bool use_shared_mem,
           py::array_t<std::uint8_t> prototypes_arr, bool use_stored_prototypes) {

            py::buffer_info img_buf = img_arr.request();

            if (img_buf.ndim != 3)
                throw std::runtime_error("Number of dimensions must be 3");
            
            auto result = py::array_t<std::uint8_t>(
                {img_buf.shape[0], img_buf.shape[1], img_buf.shape[2]});
            result[py::make_tuple(py::ellipsis())] = 0;
            py::buffer_info result_buf = result.request();

            py::buffer_info prot_buf = prototypes_arr.request();

            if (prot_buf.ndim != 2)
                throw std::runtime_error("Number of prototypes dimensions must be 2");

            cuda::k_means(
                static_cast<std::uint8_t*>(result_buf.ptr),
                static_cast<std::uint8_t*>(img_buf.ptr),
                img_buf.shape[0], img_buf.shape[1],
                k, stab_error, max_iterations,
                dev_info, use_shared_mem,
                static_cast<std::uint8_t*>(prot_buf.ptr), use_stored_prototypes
            );

            // Create an array for updated prototypes and copy the data
            std::vector<size_t> shape = { static_cast<size_t>(k), static_cast<size_t>(3) };
            auto updated_prototypes = py::array_t<std::uint8_t>(shape);
            std::memcpy(updated_prototypes.mutable_data(), prot_buf.ptr, k * 3 * sizeof(std::uint8_t));

            return py::make_tuple(result, updated_prototypes);
        });

    subm.def(
        "deinit_k_means", 
        [](cuda::KmeansInfo& dev_info) {
            cuda::deinit_k_means(dev_info);    
        });
}