/*
 * Copyright 2025 Adam Buechner
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include "adventure/dynamic_array.hpp"

namespace ad = adventure;

namespace {

struct Simple {
  int x;
  int y;
  constexpr Simple(int a = 0, int b = 0) noexcept : x(a), y(b) {}
  bool operator==(const Simple &other) const noexcept {
    return x == other.x && y == other.y;
  }
};

}  // anonymous namespace

TEST(DynamicArrayBasic, PushBackAndSize) {
  ad::internal::DynamicArray<int> v;
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(v.size(), 0u);
  EXPECT_EQ(v.capacity(), 0u);

  v.push_back(10);
  EXPECT_EQ(v.size(), 1u);
  EXPECT_GE(v.capacity(), 1u);
  EXPECT_EQ(v[0], 10);

  v.push_back(20);
  v.push_back(30);
  EXPECT_EQ(v.size(), 3u);
  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[1], 20);
  EXPECT_EQ(v[2], 30);
}

TEST(DynamicArrayGrowth, AutomaticResize) {
  ad::internal::DynamicArray<int> v;
  const std::size_t N = 1000;

  for (std::size_t i = 0; i < N; ++i) {
    v.push_back(static_cast<int>(i));
  }
  EXPECT_EQ(v.size(), N);
  for (std::size_t i = 0; i < N; ++i) {
    EXPECT_EQ(v[i], static_cast<int>(i));
  }
  // Capacity should be at least N.
  EXPECT_GE(v.capacity(), N);
}

//  emplace_back with a non-trivial (but still POD) type
TEST(DynamicArrayEmplace, StructConstruction) {
  ad::internal::DynamicArray<Simple> v;
  v.emplace_back(1, 2);
  v.emplace_back(3, 4);
  EXPECT_EQ(v.size(), 2u);
  EXPECT_EQ(v[0], Simple(1, 2));
  EXPECT_EQ(v[1], Simple(3, 4));
}

//  clear() should reset size but keep capacity
TEST(DynamicArrayClear, SizeReset) {
  ad::internal::DynamicArray<int> v;
  v.reserve(64);
  std::size_t cap_before = v.capacity();

  for (int i = 0; i < 10; ++i) v.push_back(i);
  EXPECT_EQ(v.size(), 10u);
  v.clear();
  EXPECT_EQ(v.size(), 0u);
  EXPECT_EQ(v.capacity(), cap_before);
}

//  resize() up and down
TEST(DynamicArrayResize, UpAndDown) {
  ad::internal::DynamicArray<int> v;
  v.resize(5);
  EXPECT_EQ(v.size(), 5u);
  // All newly created elements are value-initialized (zero for int).
  for (std::size_t i = 0; i < 5; ++i) EXPECT_EQ(v[i], 0);

  v[2] = 42;
  v.resize(3);
  EXPECT_EQ(v.size(), 3u);
  EXPECT_EQ(v[2], 42);

  // Growing again should default-initialize new slots.
  v.resize(6);
  EXPECT_EQ(v.size(), 6u);
  EXPECT_EQ(v[3], 0);
  EXPECT_EQ(v[4], 0);
  EXPECT_EQ(v[5], 0);
}
