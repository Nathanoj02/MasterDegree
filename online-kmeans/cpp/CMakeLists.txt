cmake_minimum_required(VERSION 3.12.2)
message(STATUS "CMake version: ${CMAKE_VERSION}")

# ------------------------------------------------------------------------------
# Setup project

# Project with not optional CUDA
project(signals LANGUAGES CUDA CXX C)
set(CMAKE_CXX_STANDARD 17)
set(DEPENDENCIES "")
set(INCLUDES "")

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(BUILD_PYTHON true CACHE BOOL "Build Python interface library.")

# ------------------------------------------------------------------------------
# Setup dependencies.
include(cmake/GPUConfig.cmake)

set(DEPENDENCIES ${DEPENDENCIES} -lstdc++fs)

# ------------------------------------------------------------------------------
# Setup CUDA.
include(cmake/GPUConfig.cmake)

find_package(CUDA REQUIRED)
set(INCLUDES ${INCLUDES} ${CUDA_INCLUDE_DIRS})
set(DEPENDENCIES ${DEPENDENCIES} ${CUDA_LIBRARIES})

# ------------------------------------------------------------------------------
# Setup OpenCV.
find_package(OpenCV REQUIRED)
set(INCLUDES ${INCLUDES} ${OpenCV_INCLUDE_DIRS})
set(DEPENDENCIES ${DEPENDENCIES} ${OpenCV_LIBS})
list(APPEND CUDA_NVCC_FLAGS ${OpenCV_COMPUTE_CAPABILITIES})
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV linking library dependecies: ${OpenCV_LIBS}")
message(STATUS "OpenCV compilation compute capability: ${OpenCV_COMPUTE_CAPABILITIES}")
message(STATUS "OpenCV compilation with CUDA version: ${OpenCV_CUDA_VERSION}")
message(STATUS "OpenCV_USE_CUBLAS: ${OpenCV_USE_CUBLAS}")
message(STATUS "OpenCV_USE_CUFFT: ${OpenCV_USE_CUFFT}")
message(STATUS "OpenCV_USE_NVCUVID: ${OpenCV_USE_NVCUVID}")

# ------------------------------------------------------------------------------
# Set the C++ compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")

# ------------------------------------------------------------------------------
# Set the C compiler options
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb")

# ------------------------------------------------------------------------------
# Set the CUDA compiler options.
set(CA ${CMAKE_CUDA_ARCHITECTURES}) # Compute Capability
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${CA},code=sm_${CA}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

# ------------------------------------------------------------------------------
# Add the source files.
set(CXX_SOURCE_FILES
    src/core.cpp
)

set(CUDA_SOURCE_FILES
    src/core_par.cu
    src/error.cu
)

# ------------------------------------------------------------------------------
# Add the library.
set(LIBRARY_NAME ${PROJECT_NAME})

set(SOURCES
    ${CXX_SOURCE_FILES}
    ${CUDA_SOURCE_FILES}
)

include_directories(include)
include_directories(${INCLUDES})

if(BUILD_PYTHON)
    message(STATUS "PYTHON interface enabled")
    add_subdirectory(python)
endif()

add_library(${LIBRARY_NAME} SHARED
    ${SOURCES}
)
target_link_libraries(${LIBRARY_NAME} ${DEPENDENCIES})

# set(BUILD_EXAMPLE true CACHE BOOL "Build examples code")
# if(BUILD_EXAMPLE)
#     add_subdirectory(example)
# endif()