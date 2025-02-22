/***************************************************************************
 *            benchmark/par/error.cuh
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/


/*! \file  error.cuh
 *  \brief 
 */

#ifndef ERROR_CUH
#define ERROR_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#define CHECK_CUDA_ERROR                                                       \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cuda_error::getLastCudaError(__FILE__, __LINE__, __func__);            \
    }

#define SAFE_CALL(function)                                                    \
    {                                                                          \
        cuda_error::safe_call(function, __FILE__, __LINE__, __func__);         \
    }



namespace cuda_error {

void cudaErrorHandler(cudaError_t error,
                      const char* error_message,
                      const char* file,
                      int         line,
                      const char* func_name);


void getLastCudaError(const char* file, int line, const char* func_name);

void safe_call(cudaError_t error,
               const char* file,
               int         line,
               const char* func_name);


} // namespace cuda_error

#endif // ERROR_CUH