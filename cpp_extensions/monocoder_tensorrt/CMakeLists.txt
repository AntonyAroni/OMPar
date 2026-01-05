cmake_minimum_required(VERSION 3.15)
project(monocoder_trt_cpp LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimizaciones
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Pybind11
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

# Rutas locales (Headers descargados)
set(LOCAL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Rutas del entorno virtual (Librerías)
set(SITE_PACKAGES "${CMAKE_CURRENT_SOURCE_DIR}/../../ompar_env/lib/python3.12/site-packages")
set(TRT_LIB_DIR "${SITE_PACKAGES}/tensorrt_libs")
set(CUDA_LIB_DIR "${SITE_PACKAGES}/nvidia/cuda_runtime/lib")

# Buscar librerías (Nombres versionados explícitos)
find_library(NVINFER_LIB NAMES libnvinfer.so.10 nvinfer PATHS ${TRT_LIB_DIR} NO_DEFAULT_PATH)
find_library(CUDART_LIB NAMES libcudart.so.12 cudart PATHS ${CUDA_LIB_DIR} /usr/local/cuda/lib64)

if (NOT NVINFER_LIB)
    message(FATAL_ERROR "libnvinfer.so.10 not found in ${TRT_LIB_DIR}")
endif()

if (NOT CUDART_LIB)
    message(WARNING "libcudart.so.12 not found in ${CUDA_LIB_DIR}, trying system default")
    find_library(CUDART_LIB cudart)
endif()

message(STATUS "NvInfer: ${NVINFER_LIB}")
message(STATUS "Cudart: ${CUDART_LIB}")

# Module
pybind11_add_module(monocoder_trt_cpp
    bindings.cpp
    monocoder_trt.cpp
)

target_include_directories(monocoder_trt_cpp PRIVATE
    ${LOCAL_INCLUDE_DIR}
    ${SITE_PACKAGES}/nvidia/cuda_runtime/include
    /usr/local/cuda/include
)

target_link_libraries(monocoder_trt_cpp PRIVATE
    ${NVINFER_LIB}
    ${CUDART_LIB}
)

# RPATH para encontrar librerías en runtime (IMPORTANTE)
set_target_properties(monocoder_trt_cpp PROPERTIES
    BUILD_RPATH "${TRT_LIB_DIR};${CUDA_LIB_DIR}"
    INSTALL_RPATH "${TRT_LIB_DIR};${CUDA_LIB_DIR}"
)

install(TARGETS monocoder_trt_cpp
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
