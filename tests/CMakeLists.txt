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

add_executable(test_ad test_ad.cpp)
target_link_libraries(test_ad PRIVATE adventure GTest::gtest_main)
add_test(NAME test_ad COMMAND test_ad)

add_executable(test_assignment test_assignment.cpp)
target_link_libraries(test_assignment PRIVATE adventure GTest::gtest_main)
add_test(NAME test_assignment COMMAND test_assignment)

add_executable(test_variable_constructors test_variable_constructors.cpp)
target_link_libraries(test_variable_constructors PRIVATE adventure GTest::gtest_main)
add_test(NAME test_variable_constructors COMMAND test_variable_constructors)

add_executable(test_ops test_ops.cpp)
target_link_libraries(test_ops PRIVATE adventure GTest::gtest_main)
add_test(NAME test_ops COMMAND test_ops)

add_executable(test_parent_count test_parent_count.cpp)
target_link_libraries(test_parent_count PRIVATE adventure GTest::gtest_main)
add_test(NAME test_parent_count COMMAND test_parent_count)

add_executable(test_constant_inactive test_constant_inactive.cpp)
target_link_libraries(test_constant_inactive PRIVATE adventure GTest::gtest_main)
add_test(NAME test_constant_inactive COMMAND test_constant_inactive)

add_executable(test_custom_node test_custom_node.cpp)
target_link_libraries(test_custom_node PRIVATE adventure GTest::gtest_main)
add_test(NAME test_custom_node COMMAND test_custom_node)

add_executable(test_thread_isolation test_thread_isolation.cpp)
target_link_libraries(test_thread_isolation PRIVATE adventure GTest::gtest_main)
add_test(NAME test_thread_isolation COMMAND test_thread_isolation)

add_executable(test_exception_safety test_exception_safety.cpp)
target_link_libraries(test_exception_safety PRIVATE adventure GTest::gtest_main)
add_test(NAME test_exception_safety COMMAND test_exception_safety)

add_executable(test_zero_derivative_pruning test_zero_derivative_pruning.cpp)
target_link_libraries(test_zero_derivative_pruning PRIVATE adventure GTest::gtest_main)
add_test(NAME test_zero_derivative_pruning COMMAND test_zero_derivative_pruning)

add_executable(test_multiple_expressions test_multiple_expressions.cpp)
target_link_libraries(test_multiple_expressions PRIVATE adventure GTest::gtest_main)
add_test(NAME test_multiple_expressions COMMAND test_multiple_expressions)

add_executable(test_dynamic_array test_dynamic_array.cpp)
target_link_libraries(test_dynamic_array PRIVATE adventure GTest::gtest_main)
add_test(NAME test_dynamic_array COMMAND test_dynamic_array)
