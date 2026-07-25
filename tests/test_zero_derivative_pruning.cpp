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
  ad::clear_tape<Scalar>();

  // Two inputs: a < b, so min(a,b) = a and d(min)/db = 0
  ad::Variable<Scalar> a(Scalar(1.0));
  ad::Variable<Scalar> b(Scalar(2.0));
  ad::register_input(a);
  ad::register_input(b);

  // y = min(a,b) - derivative w.r.t. b should be zero and thus pruned
  ad::Variable<Scalar> y = min(a, b);
  ad::register_output(y);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();

  // The node for `y` should have arity 1 (only parent a) because the
  // coefficient for b is zero and should have been pruned.
  const auto &tape = ad::get_tape<Scalar>();
  std::size_t node_idx = y.tape_index();
  EXPECT_EQ(tape.arities[node_idx], 1u);

  // Verify gradients: a gets 1, b gets 0
  EXPECT_EQ(a.grad(), Scalar(1));
  EXPECT_EQ(b.grad(), Scalar(0));
}

TEST(ZeroDerivativePruning, ComplexExpressionAllZero) {
  using Scalar = double;
  ad::clear_tape<Scalar>();

  // Two active inputs: a < b, so min(a,b) = a and d(min)/db = 0
  ad::Variable<Scalar> a(Scalar(1.0));
  ad::Variable<Scalar> b(Scalar(2.0));
  ad::register_input(a);
  ad::register_input(b);

  // Build a more complex expression:
  //   y = (min(a, b) + max(a, b)) * 0
  // Multiplication by zero makes total derivative zero.
  ad::Variable<Scalar> y = (min(a, b) + max(a, b)) * Scalar(0.0);
  ad::register_output(y);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();

  const auto &tape = ad::get_tape<Scalar>();
  std::size_t node_idx = y.tape_index();
  EXPECT_EQ(tape.arities[node_idx], 0u)
      << "All derivative coefficients should be pruned";

  EXPECT_EQ(a.grad(), Scalar(0));
  EXPECT_EQ(b.grad(), Scalar(0));
}
