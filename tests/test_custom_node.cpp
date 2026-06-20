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

TEST(CustomNode, MetadataVerification) {
  using Scalar = double;
  ad::clear_tape<Scalar>();

  ad::Variable<Scalar> x(Scalar(1.0));
  ad::Variable<Scalar> y(Scalar(2.0));
  ad::Variable<Scalar> z(Scalar(3.0));
  ad::register_input(x);
  ad::register_input(y);
  ad::register_input(z);

  // Expression with three distinct parents: x*y + sin(z)
  ad::Variable<Scalar> f = x * y + sin(z);
  // Capture the index of the expression node before registering as output.
  std::size_t expr_idx = f.tape_index();
  ad::register_output(f);
  f.grad() = Scalar(1);
  ad::backward<Scalar>();

  // The node created by materialise should be a node with arity 3.
  const auto &tape = ad::get_tape<Scalar>();
  EXPECT_EQ(tape.arities[expr_idx], 3u);

  // Compute the start offset of this node by summing arities of previous nodes
  std::size_t off = 0;
  for (std::size_t i = 0; i < expr_idx; ++i) {
    off += tape.arities[i];
  }
  std::size_t p0 = tape.edges[off].parent;
  std::size_t p1 = tape.edges[off + 1].parent;
  std::size_t p2 = tape.edges[off + 2].parent;
  std::size_t expected[3] = {x.tape_index(), y.tape_index(), z.tape_index()};
  EXPECT_NE(std::find(std::begin(expected), std::end(expected), p0),
            std::end(expected));
  EXPECT_NE(std::find(std::begin(expected), std::end(expected), p1),
            std::end(expected));
  EXPECT_NE(std::find(std::begin(expected), std::end(expected), p2),
            std::end(expected));
}
