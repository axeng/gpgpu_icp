cmake_minimum_required(VERSION 3.16)
project(gpgpu_icp)

set(CMAKE_CXX_STANDARD 17)

set(CPU_TARGET cpu)
set(GPU_TARGET gpu)

# Automatic Compilation Flag to Release
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

############### COMPILATION FLAGS ###############
# General compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -std=c++17")

# Debug compilation flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -fno-inline -fconcepts-diagnostics-depth=2 -Werror")

# Release compilation flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -flto")
#################################################

############### EXTERNAL LIBRARIES ##############
# Boost
find_package(Boost COMPONENTS system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Benchmark
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(vendor/google/benchmark EXCLUDE_FROM_ALL)
#################################################

#################### SOURCES ####################
# CPU sources
set(CPU_PARSER_SOURCES
        src/cpu/parser/parser.cc)

set(CPU_UTILS_SOURCES
        src/cpu/utils/utils.cc)

set(CPU_SOURCES
        ${CPU_PARSER_SOURCES}
        ${CPU_UTILS_SOURCES})

set(CPU_EXEC_SOURCES
        ${CPU_SOURCES}
        src/cpu/cpu.cc)

# Benchmarking Sources
set(BENCH_SOURCES
        bench/main.cc
        ${CPU_SOURCES})
#################################################

# CPU Executable
add_executable(${CPU_TARGET}
        ${CPU_EXEC_SOURCES})

# Benchmarking Executable
add_executable(bench EXCLUDE_FROM_ALL ${BENCH_SOURCES})
target_include_directories(bench PRIVATE bench)
target_link_libraries(bench PRIVATE benchmark::benchmark ${Boost_LIBRARIES})