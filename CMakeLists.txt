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
# TODO
#################################################

#################### SOURCES ####################
# CPU sources
set(CPU_PARSER_SOURCES
        src/cpu/parser/parser.cc)

set(CPU_UTILS_SOURCES
        src/cpu/utils/utils.cc)

set(CPU_SOURCES
        ${CPU_PARSER_SOURCES}
        ${CPU_UTILS_SOURCES}
        src/cpu/cpu.cc)
#################################################

# CPU Executable
add_executable(${CPU_TARGET}
        ${CPU_SOURCES})