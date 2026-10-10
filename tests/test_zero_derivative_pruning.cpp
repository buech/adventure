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

TEST(ZeroDerivativePruning, MinOpEdgePruned) {
  using Scalar = double;
  auto &tape = ad::get_tape<Scalar>();
  tape.clear();

  // Two inputs: a < b, so min(a,b) = a and d(min)/db = 0
  ad::Variable<Scalar> a(Scalar(1.0));
  ad::Variable<Scalar> b(Scalar(2.0));
  tape.register_input(a);
  tape.register_input(b);

  // y = min(a,b) - derivative w.r.t. b should be zero and thus pruned
  ad::Variable<Scalar> y = min(a, b);
  // Capture the MIN node index before register_output wraps it in a COPY node.
  const std::size_t node_idx = y.tape_index();
  tape.register_output(y);
  y.grad() = Scalar(1);
  tape.backward();

  // The node for `y` is a MIN node with both operands recorded; the
  // pruning happens dynamically during the backward sweep, which routes
  // the seed only to the winning operand.
  EXPECT_EQ(tape.ops[node_idx], ad::OpCode::MIN);
  EXPECT_EQ(tape.op_arity(tape.ops[node_idx]), 2u);

  // Verify gradients: a gets 1, b gets 0
  EXPECT_EQ(a.grad(), Scalar(1));
  EXPECT_EQ(b.grad(), Scalar(0));
}

TEST(ZeroDerivativePruning, ComplexExpressionAllZero) {
  using Scalar = double;
  auto &tape = ad::get_tape<Scalar>();
  tape.clear();

  // Two active inputs: a < b, so min(a,b) = a and d(min)/db = 0
  ad::Variable<Scalar> a(Scalar(1.0));
  ad::Variable<Scalar> b(Scalar(2.0));
  tape.register_input(a);
  tape.register_input(b);

  // Build a more complex expression:
  //   y = (min(a, b) + max(a, b)) * 0
  // Multiplication by zero makes total derivative zero.
  ad::Variable<Scalar> y = (min(a, b) + max(a, b)) * Scalar(0.0);
  // Capture the MUL node index before register_output wraps it in a COPY node.
  const std::size_t node_idx = y.tape_index();
  tape.register_output(y);
  y.grad() = Scalar(1);
  tape.backward();

  EXPECT_EQ(tape.ops[node_idx], ad::OpCode::MUL)
      << "All derivative coefficients should be pruned";

  // The multiplier is the constant 0, so the backward sweep contributes
  // nothing to either input.
  EXPECT_EQ(a.grad(), Scalar(0));
  EXPECT_EQ(b.grad(), Scalar(0));
}
