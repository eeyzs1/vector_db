
cmake_minimum_required(VERSION 3.18)
project(vectordb_nanobind)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
endif()

find_package(Python3 COMPONENTS Interpreter Development)

include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v1.8.0
)
FetchContent_MakeAvailable(nanobind)

add_library(vectordb_core INTERFACE)
target_include_directories(vectordb_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/core
    ${Python3_INCLUDE_DIRS}
)

set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../python/vectordb/cpp)

function(add_nanobind_module name binding_file source_dir)
    nanobind_add_module(${name}
        ${binding_file}
        ../cpp/algorithms/${source_dir}/${source_dir}.cpp
    )
    target_link_libraries(${name} PRIVATE vectordb_core)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/core
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/algorithms/${source_dir}
    )
    set_target_properties(${name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    )
endfunction()

add_nanobind_module(_flat_nanobind bindings_flat.cpp flat)
add_nanobind_module(_flat_ip_nanobind bindings_flat_ip.cpp flat_ip)
add_nanobind_module(_ivf_nanobind bindings_ivf.cpp ivf)
add_nanobind_module(_hnsw_nanobind bindings_hnsw.cpp hnsw)
add_nanobind_module(_pq_nanobind bindings_pq.cpp pq)
add_nanobind_module(_lsh_nanobind bindings_lsh.cpp lsh)
add_nanobind_module(_kdtree_nanobind bindings_kdtree.cpp kdtree)
add_nanobind_module(_balltree_nanobind bindings_balltree.cpp balltree)
add_nanobind_module(_annoy_nanobind bindings_annoy.cpp annoy)

