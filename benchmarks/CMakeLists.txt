# Copyright (c) 2025 Adam Buechner
#
# Licensed under the Apache License 2.0, or, at your option, the MIT License.
#
# SPDX-License-Identifier: Apache-2.0 OR MIT

add_executable(benchmark_ad benchmark_ad.cpp)
target_link_libraries(benchmark_ad PRIVATE adventure benchmark::benchmark)
