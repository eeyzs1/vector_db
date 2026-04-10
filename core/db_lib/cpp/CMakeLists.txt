# 模块化编译的 CMake 配置

cmake_minimum_required(VERSION 3.18)
project(vectordb)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_OPENBLAS "Enable OpenBLAS for matrix operations" ON)
option(BUILD_TESTS "Build unit tests with GoogleTest" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
endif()

find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

if(USE_OPENBLAS)
    find_package(PkgConfig)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(OPENBLAS openblas)
        if(OPENBLAS_FOUND)
            add_definitions(-DUSE_OPENBLAS)
            message(STATUS "OpenBLAS found, enabling BLAS optimizations")
        else()
            message(STATUS "OpenBLAS not found, building without BLAS support")
            set(USE_OPENBLAS OFF)
        endif()
    else()
        message(STATUS "pkg-config not found, building without BLAS support")
        set(USE_OPENBLAS OFF)
    endif()
endif()

add_library(vectordb_core INTERFACE)
target_include_directories(vectordb_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${pybind11_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
)
if(USE_OPENBLAS)
    target_link_libraries(vectordb_core INTERFACE ${OPENBLAS_LIBRARIES})
endif()

pybind11_add_module(_ivf
    algorithms/ivf/ivf.cpp
    algorithms/ivf/bindings.cpp
)
target_link_libraries(_ivf PRIVATE
    vectordb_core
)
target_include_directories(_ivf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/ivf
)

pybind11_add_module(_hnsw
    algorithms/hnsw/hnsw.cpp
    algorithms/hnsw/bindings.cpp
)
target_link_libraries(_hnsw PRIVATE
    vectordb_core
)
target_include_directories(_hnsw PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/hnsw
)

pybind11_add_module(_pq
    algorithms/pq/pq.cpp
    algorithms/pq/bindings.cpp
)
target_link_libraries(_pq PRIVATE
    vectordb_core
)
target_include_directories(_pq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/pq
)

pybind11_add_module(_lsh
    algorithms/lsh/lsh.cpp
    algorithms/lsh/bindings.cpp
)
target_link_libraries(_lsh PRIVATE
    vectordb_core
)
target_include_directories(_lsh PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/lsh
)

pybind11_add_module(_kdtree
    algorithms/kdtree/kdtree.cpp
    algorithms/kdtree/bindings.cpp
)
target_link_libraries(_kdtree PRIVATE
    vectordb_core
)
target_include_directories(_kdtree PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/kdtree
)

pybind11_add_module(_balltree
    algorithms/balltree/balltree.cpp
    algorithms/balltree/bindings.cpp
)
target_link_libraries(_balltree PRIVATE
    vectordb_core
)
target_include_directories(_balltree PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/balltree
)

pybind11_add_module(_flat
    algorithms/flat/flat.cpp
    algorithms/flat/bindings.cpp
)
target_link_libraries(_flat PRIVATE
    vectordb_core
)
target_include_directories(_flat PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/flat
)

pybind11_add_module(_flat_ip
    algorithms/flat_ip/flat_ip.cpp
    algorithms/flat_ip/bindings.cpp
)
target_link_libraries(_flat_ip PRIVATE
    vectordb_core
)
target_include_directories(_flat_ip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/flat_ip
)

pybind11_add_module(_annoy
    algorithms/annoy/annoy.cpp
    algorithms/annoy/bindings.cpp
)
target_link_libraries(_annoy PRIVATE
    vectordb_core
)
target_include_directories(_annoy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/annoy
)

set_target_properties(_ivf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/vectordb/cpp
)

set_target_properties(_hnsw PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/vectordb/cpp
)

set_target_properties(_pq PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/vectordb/cpp
)

set_target_properties(_lsh PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/vectordb/cpp
)

set_target_properties(_kdtree PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/vectordb/cpp
)

set_target_properties(_balltree PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/vectordb/cpp
)

set_target_properties(_flat PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/vectordb/cpp
)

set_target_properties(_flat_ip PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/vectordb/cpp
)

set_target_properties(_annoy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/vectordb/cpp
)

# GoogleTest configuration
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    # Create test directory if it doesn't exist
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)

    # Add test executable for Flat L2
    add_executable(test_flat_l2
        tests/test_flat_l2.cpp
        algorithms/flat/flat.cpp
    )
    target_link_libraries(test_flat_l2
        GTest::gtest_main
        vectordb_core
    )
    target_include_directories(test_flat_l2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/flat
    )

    include(GoogleTest)
    gtest_discover_tests(test_flat_l2)
endif()
