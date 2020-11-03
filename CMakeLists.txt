cmake_minimum_required(VERSION 3.16)
project(gpgpu_icp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CPU_TARGET cpu)
set(GPU_FULL_TARGET gpu_full)
set(BENCH_TARGET bench)
set(UNIT_TESTS_TARGET unit_tests)

# Automatic Compilation Flag to Release
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

############# CXX COMPILATION FLAGS #############
# General compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -pedantic -std=c++17")

# Debug compilation flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -lineinfo")

# Release compilation flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -flto")
#################################################

############# CUDA COMPILATION FLAGS ############
# General compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

# Debug compilation flags
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -g3 -lineinfo")

# Release compilation flags
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
#################################################

############### EXTERNAL LIBRARIES ##############
# CUDA
find_package(CUDA REQUIRED)

# Benchmark
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(vendor/google/benchmark EXCLUDE_FROM_ALL)
#################################################

################## DIRECTORIES ##################
include_directories(src)
#################################################

#################### SOURCES ####################
# CPU sources
set(CPU_ICP_SOURCES
        src/cpu/icp/icp.cc)

set(CPU_PARSER_SOURCES
        src/cpu/parser/parser.cc)

set(CPU_UTILS_SOURCES
        src/cpu/utils/utils.cc
        src/cpu/utils/lib-matrix.cc
        src/cpu/utils/matrix.cc)

set(CPU_SOURCES
        ${CPU_ICP_SOURCES}
        ${CPU_PARSER_SOURCES}
        ${CPU_UTILS_SOURCES})

set(CPU_EXEC_SOURCES
        ${CPU_SOURCES}
        src/cpu/cpu.cc)

# GPU full sources
set(GPU_FULL_ICP_SOURCES
        src/gpu_full/icp/icp.cu)

set(GPU_FULL_PARSER_SOURCES
        src/gpu_full/parser/parser.cu)

set(GPU_FULL_UTILS_SOURCES
        src/gpu_full/utils/utils.cu
        src/gpu_full/utils/lib-matrix.cu
        src/gpu_full/utils/matrix.cu)

set(GPU_FULL_SOURCES
        ${GPU_FULL_ICP_SOURCES}
        ${GPU_FULL_PARSER_SOURCES}
        ${GPU_FULL_UTILS_SOURCES})

set(GPU_FULL_EXEC_SOURCES
        ${GPU_FULL_SOURCES}
        src/gpu_full/gpu.cu)

# Benchmarking Sources
set(BENCH_SOURCES
        bench/main.cu
        ${CPU_SOURCES}
        ${CPU_FULL_SOURCES})

# Unit Tests
set(UNIT_TESTS_SOURCES
        tests/unit_tests/main.cc
        ${CPU_SOURCES})
#################################################

# CPU Executable
add_executable(${CPU_TARGET}
        ${CPU_EXEC_SOURCES})

add_executable(${GPU_TEST_TARGET}
        ${GPU_TEST_EXEC_SOURCES})

add_executable(${GPU_FULL_TARGET}
        ${GPU_FULL_EXEC_SOURCES})

# Benchmarking Executable
add_executable(${BENCH_TARGET} EXCLUDE_FROM_ALL ${BENCH_SOURCES})
target_include_directories(bench PRIVATE bench)
target_link_libraries(bench PRIVATE benchmark::benchmark)

###############################################################################
#                                 GOOGLE TEST                                 #
###############################################################################

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif ()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif ()

# Unit Tests Executable
add_executable(${UNIT_TESTS_TARGET} ${UNIT_TESTS_SOURCES})
target_link_libraries(${UNIT_TESTS_TARGET} gtest_main)
