cmake_minimum_required(VERSION 3.16)
project(gpgpu_icp)

set(CMAKE_CXX_STANDARD 17)

# Automatic Compilation Flag to Release
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

############### COMPILATION FLAGS ###############
# General compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -std=c++20")

# Debug compilation flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -fno-inline -fconcepts-diagnostics-depth=2 -Werror")

# Release compilation flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -flto")
#################################################

############### EXTERNAL LIBRARIES ##############
# TODO
#################################################

#################### SOURCES ####################
# TODO
#################################################

# TODO