# Copyright (c) 2025 Adam Buechner
#
# Licensed under the Apache License 2.0, or, at your option, the MIT License.
#
# SPDX-License-Identifier: Apache-2.0 OR MIT

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
option(BUILD_DOCS "Build documentation" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
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

        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/
                DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/adventure)
    else()
        message(STATUS "Doxygen not found, documentation target 'doc' will not be created")
    endif()
endif()

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
if(BUILD_TESTS)
    include(FetchContent)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
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
endif()

# Benchmarks (optional)
if(BUILD_BENCHMARKS)
    if(NOT BUILD_TESTS)
        include(FetchContent)
    endif()
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
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

# Install README, LICENSE and NOTICE files
install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE-APACHE"
        "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE-MIT"
        "${CMAKE_CURRENT_SOURCE_DIR}/NOTICE"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/adventure)

# Export the target set for the install tree
install(EXPORT adventureTargets
        FILE adventureTargets.cmake
        NAMESPACE adventure::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adventure)
