/*
 * Copyright (c) 2025 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#include <gtest/gtest.h>

#include "adventure/expr.hpp"
#include "adventure/tape.hpp"
#include "adventure/variable.hpp"

namespace ad = adventure;

TEST(CustomNode, MetadataVerification) {
  using Scalar = double;
  auto &tape = ad::get_tape<Scalar>();
  tape.clear();

  ad::Variable<Scalar> x(Scalar(1.0));
  ad::Variable<Scalar> y(Scalar(2.0));
  ad::Variable<Scalar> z(Scalar(3.0));
  tape.register_input(x);
  tape.register_input(y);
  tape.register_input(z);

  // Expression with three distinct parents: x*y + sin(z)
  ad::Variable<Scalar> f = x * y + sin(z);
  // Capture the index of the expression node before registering as output.
  std::size_t expr_idx = f.tape_index();
  tape.register_output(f);
  f.grad() = Scalar(1);
  tape.backward();

  // The node created by materialise should be a node with arity 3.
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
