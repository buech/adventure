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
#include "adventure/tape.hpp"
#include "adventure/variable.hpp"

namespace ad = adventure;

TEST(MultipleExpressions, VariableReuse) {
  using Scalar = double;

  // 1. Fresh tape and a single input variable.
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(3.0));
  ad::register_input(x);  // x becomes leaf node

  // 2. First expression: y = x * x
  ad::Variable<Scalar> y = x * x;  // binary node

  // 3. Second expression that re-uses y: z = y + x
  ad::Variable<Scalar> z = y + x;  // another binary node

  // Seed the output and run backward.
  ad::register_output(z);
  z.grad() = Scalar(1);
  ad::backward<Scalar>();

  // 4. Verify gradients.
  // dz/dx = 2*x + 1  (x = 3 -> 7)
  EXPECT_DOUBLE_EQ(x.grad(), Scalar(2) * Scalar(3.0) + Scalar(1));

  // dz/dy = 1 (the seed propagates unchanged to y)
  EXPECT_DOUBLE_EQ(y.grad(), Scalar(1));

  // 5. Verify tape size.
  // Expected nodes: (1) leaf (x), (2) y = x*x, (3) z = y+x, (4) unary after
  // register_output(z)
  EXPECT_EQ(ad::get_tape<Scalar>().size(), 4u);
}
