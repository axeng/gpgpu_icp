cmake_minimum_required(VERSION 3.16)
project(gpgpu_icp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CPU_TARGET cpu)
set(GPU_1_TARGET gpu_1)
set(GPU_2_TARGET gpu_2)
set(GPU_3_TARGET gpu_3)
set(GPU_FINAL_TARGET gpu_final)
set(BENCH_TARGET bench)

# Automatic Compilation Flag to Release
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

############# CXX COMPILATION FLAGS #############
# General compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -pedantic -std=c++17")

# Debug compilation flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -g -pg")

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

# GPU 1 sources
set(GPU_1_ICP_SOURCES
        src/gpu_1/icp/icp.cu)

set(GPU_1_PARSER_SOURCES
        src/gpu_1/parser/parser.cu)

set(GPU_1_UTILS_SOURCES
        src/gpu_1/utils/utils.cu
        src/gpu_1/utils/lib-matrix.cu
        src/gpu_1/utils/matrix.cu)

set(GPU_1_SOURCES
        ${GPU_1_ICP_SOURCES}
        ${GPU_1_PARSER_SOURCES}
        ${GPU_1_UTILS_SOURCES})

set(GPU_1_EXEC_SOURCES
        ${GPU_1_SOURCES}
        src/gpu_1/gpu.cu)

# GPU 2 sources
set(GPU_2_ICP_SOURCES
        src/gpu_2/icp/icp.cu)

set(GPU_2_PARSER_SOURCES
        src/gpu_2/parser/parser.cu)

set(GPU_2_UTILS_SOURCES
        src/gpu_2/utils/utils.cu
        src/gpu_2/utils/lib-matrix.cu
        src/gpu_2/utils/matrix.cu)

set(GPU_2_SOURCES
        ${GPU_2_ICP_SOURCES}
        ${GPU_2_PARSER_SOURCES}
        ${GPU_2_UTILS_SOURCES})

set(GPU_2_EXEC_SOURCES
        ${GPU_2_SOURCES}
        src/gpu_2/gpu.cu)

# GPU 3 sources
set(GPU_3_ICP_SOURCES
        src/gpu_3/icp/icp.cu)

set(GPU_3_PARSER_SOURCES
        src/gpu_3/parser/parser.cu)

set(GPU_3_UTILS_SOURCES
        src/gpu_3/utils/utils.cu
        src/gpu_3/utils/lib-matrix.cu
        src/gpu_3/utils/matrix.cu)

set(GPU_3_SOURCES
        ${GPU_3_ICP_SOURCES}
        ${GPU_3_PARSER_SOURCES}
        ${GPU_3_UTILS_SOURCES})

set(GPU_3_EXEC_SOURCES
        ${GPU_3_SOURCES}
        src/gpu_3/gpu.cu)

# GPU final sources
set(GPU_FINAL_ICP_SOURCES
        src/gpu_final/icp/icp.cu)

set(GPU_FINAL_PARSER_SOURCES
        src/gpu_final/parser/parser.cu)

set(GPU_FINAL_UTILS_SOURCES
        src/gpu_final/utils/utils.cu
        src/gpu_final/utils/lib-matrix.cu
        src/gpu_final/utils/matrix.cu)

set(GPU_FINAL_SOURCES
        ${GPU_FINAL_ICP_SOURCES}
        ${GPU_FINAL_PARSER_SOURCES}
        ${GPU_FINAL_UTILS_SOURCES})

set(GPU_FINAL_EXEC_SOURCES
        ${GPU_FINAL_SOURCES}
        src/gpu_final/gpu.cu)

# Benchmarking Sources
set(BENCH_SOURCES
        bench/main.cu
        ${CPU_SOURCES}
        ${GPU_1_SOURCES}
        ${GPU_2_SOURCES}
        ${GPU_3_SOURCES}
        ${GPU_FINAL_SOURCES})
#################################################

# CPU Executable
add_executable(${CPU_TARGET}
        ${CPU_EXEC_SOURCES})

add_executable(${GPU_1_TARGET}
        ${GPU_1_EXEC_SOURCES})

add_executable(${GPU_2_TARGET}
        ${GPU_2_EXEC_SOURCES})

add_executable(${GPU_3_TARGET}
        ${GPU_3_EXEC_SOURCES})

add_executable(${GPU_FINAL_TARGET}
        ${GPU_FINAL_EXEC_SOURCES})



# Benchmarking Executable
add_executable(${BENCH_TARGET} EXCLUDE_FROM_ALL ${BENCH_SOURCES})
target_include_directories(bench PRIVATE bench)
target_link_libraries(bench PRIVATE benchmark::benchmark)