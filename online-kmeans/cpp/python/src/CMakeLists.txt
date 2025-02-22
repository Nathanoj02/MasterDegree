set(LIBRARY_NAME pysignals-module-obj)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(SYSTEM ${PYBIND11_INCLUDE_DIR})

message(STATUS "Please note that you need to use the Python 3 interpreter to import pysignals correctly.")

add_library(${LIBRARY_NAME} OBJECT
    sequential_submodule.cpp
    parallel_submodule.cpp
    signals_module.cpp)
set_target_properties(
    ${LIBRARY_NAME} PROPERTIES 
        COMPILE_FLAGS 
            "-fPIC"
)
