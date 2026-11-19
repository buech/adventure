# Copyright (c) 2025 Adam Buechner
#
# Licensed under the Apache License 2.0, or, at your option, the MIT License.
#
# SPDX-License-Identifier: Apache-2.0 OR MIT

# Enable useful compiler warnings and disable -Wself-assign-overloaded because
# of the self-assignment test in test_assignment.cpp
add_compile_options(-Wall -Wpedantic -Wextra -Wno-self-assign-overloaded)

file(GLOB test_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp"
)

foreach(test_source IN LISTS test_sources)
    get_filename_component(test_name "${test_source}" NAME_WE)

    add_executable("${test_name}" "${test_source}")
    target_link_libraries("${test_name}" PRIVATE adventure GTest::gtest_main)
    add_test(NAME "${test_name}" COMMAND "${test_name}")
endforeach()
