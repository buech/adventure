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

#include "adventure/expr.hpp"
#include "adventure/variable.hpp"

namespace ad = adventure;

TEST(ConstantExpression, InactiveVariable) {
  // A pure constant expression should produce an inactive Variable.
  auto c = ad::materialise(ad::ConstExpr<double>(5.0));
  EXPECT_EQ(c.tape_index(), ad::Variable<double>::invalid_idx);
  EXPECT_DOUBLE_EQ(c.value(), 5.0);
}
