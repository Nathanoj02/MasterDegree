/***************************************************************************
 *            error.cpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/


#include "error.cuh"


namespace cuda_error {

void cudaErrorHandler(cudaError_t error,
                      const char* error_message,
                      const char* file,
                      int         line,
                      const char* func_name) {
    if (cudaSuccess != error) {
        std::cerr << "\nCUDA error\n" << file << "(" << line << ")"
                  << " [ " << func_name << " ] : " << error_message
                  << " -> " << cudaGetErrorString(error)
                  << "(" << static_cast<int>(error) << ")\n"
                  << std::endl;
        assert(false);                                                  //NOLINT
        std::atexit(reinterpret_cast<void(*)()>(cudaDeviceReset));
        std::exit(EXIT_FAILURE);
    }
}

void getLastCudaError(const char* file, int line, const char* func_name) {
    cudaErrorHandler(cudaGetLastError(), "", file, line, func_name);
}

void safe_call(cudaError_t error,
               const char* file,
               int         line,
               const char* func_name) {
    cudaErrorHandler(error, "", file, line, func_name);
}


} // namespace cuda_error