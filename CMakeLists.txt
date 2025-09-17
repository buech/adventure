# Copyright 2025 Adam Buechner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18)
project(adventure LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Disable GNU extensions to the C++ standard
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable useful compiler warnings
add_compile_options(-Wall -Wpedantic)

# Generate compile_commands.json for tooling support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(GNUInstallDirs)

# Doxygen documentation
find_package(Doxygen REQUIRED)

# Configure Doxygen input file
set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

# Add a custom target to generate the documentation
add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)

# Header-only interface library
add_library(adventure INTERFACE)
target_include_directories(adventure INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Export the INTERFACE target for package use
export(TARGETS adventure
       FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/adventureTargets.cmake"
       NAMESPACE adventure::)

# Create package version file
include(CMakePackageConfigHelpers)
set(adventure_VERSION 0.0.1)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/adventureConfigVersion.cmake"
    VERSION ${adventure_VERSION}
    COMPATIBILITY SameMajorVersion)

# Configure the package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/adventureConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/adventureConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/adventure")

# GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    )
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)

# Benchmarks (optional)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING NO)
    FetchContent_Declare(
        benchmark
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        URL https://github.com/google/benchmark/archive/refs/tags/v1.9.5.zip
        )
    FetchContent_MakeAvailable(benchmark)
    add_subdirectory(benchmarks)
endif()

# Optional install step

# Install public headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")

# Install the INTERFACE target (so the exported target can be imported after install)
install(TARGETS adventure
        EXPORT adventureTargets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install the generated package files
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/cmake/adventureConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake/adventureConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adventure)

# Export the target set for the install tree
install(EXPORT adventureTargets
        FILE adventureTargets.cmake
        NAMESPACE adventure::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adventure)
