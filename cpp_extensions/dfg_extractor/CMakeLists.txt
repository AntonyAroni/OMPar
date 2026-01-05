cmake_minimum_required(VERSION 3.15)
project(dfg_extractor_cpp LANGUAGES CXX C)

# C++17 requerido
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimizaciones y PIC para shared libraries
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -fPIC")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Buscar pybind11 usando Python
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

# Buscar OpenMP
find_package(OpenMP REQUIRED)

# Directorios
set(TREE_SITTER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tree-sitter-lib")
set(PARSER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../parser")

# Compilar tree-sitter como librería estática
add_library(tree_sitter STATIC
    ${TREE_SITTER_DIR}/lib/src/lib.c
)
target_include_directories(tree_sitter PUBLIC
    ${TREE_SITTER_DIR}/lib/include
    ${TREE_SITTER_DIR}/lib/src
)

# Compilar parser C# (para C)
add_library(tree_sitter_csharp STATIC
    ${PARSER_DIR}/vendor/tree-sitter-c-sharp/src/parser.c
    ${PARSER_DIR}/vendor/tree-sitter-c-sharp/src/scanner.c
)
target_include_directories(tree_sitter_csharp PUBLIC
    ${PARSER_DIR}/vendor/tree-sitter-c-sharp/src
    ${TREE_SITTER_DIR}/lib/include
)

# Crear módulo Python
pybind11_add_module(dfg_extractor_cpp
    bindings.cpp
    dfg_extractor.cpp
)

# Incluir directorios
target_include_directories(dfg_extractor_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TREE_SITTER_DIR}/lib/include
)

# Linkear librerías
target_link_libraries(dfg_extractor_cpp PRIVATE
    tree_sitter
    tree_sitter_csharp
    OpenMP::OpenMP_CXX
)

# Propiedades del target
set_target_properties(dfg_extractor_cpp PROPERTIES
    PREFIX ""
    OUTPUT_NAME "dfg_extractor_cpp"
    SUFFIX ".so"
)

# Instalar en el directorio raíz del proyecto
install(TARGETS dfg_extractor_cpp
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Mensaje de configuración
message(STATUS "DFG Extractor C++ Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "  Tree-sitter: ${TREE_SITTER_DIR}")
message(STATUS "  Parser: ${PARSER_DIR}")
