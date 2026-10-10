/*
 * Copyright (c) 2025 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

#include "adventure/expr.hpp"
#include "adventure/tape.hpp"
#include "adventure/variable.hpp"

namespace ad = adventure;

namespace {

/// Offset of `node`'s first operand within the flat operand buffer.
/**
 * Operands are stored contiguously in record order, so the start offset of a
 * node is the sum of the arities of all preceding nodes (mirroring the cursor
 * walk performed by Tape::backward).
 */
std::size_t operand_offset(const ad::Tape<double> &tape, ad::index_t node) {
  std::size_t off = 0;
  for (ad::index_t i = 0; i < node; ++i) off += tape.op_arity(tape.ops[i]);
  return off;
}

}  // namespace

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

  // The node created by materialise is an ADD node with two operands
  // (x*y and sin(z)); together with their sub-operands the expression
  // touches all three inputs.
  EXPECT_EQ(tape.ops[expr_idx], ad::OpCode::ADD);
  EXPECT_EQ(tape.op_arity(tape.ops[expr_idx]), 2u);

  // The two direct operands of the ADD node.
  const std::size_t off = operand_offset(tape, expr_idx);
  const std::size_t mul_idx = tape.operands[off];
  const std::size_t sin_idx = tape.operands[off + 1];

  // x*y is a MUL node whose parents are x and y.
  EXPECT_EQ(tape.ops[mul_idx], ad::OpCode::MUL);
  const std::size_t mul_off = operand_offset(tape, mul_idx);
  const std::size_t p0 = tape.operands[mul_off];
  const std::size_t p1 = tape.operands[mul_off + 1];

  // sin(z) is a SIN node whose parent is z.
  EXPECT_EQ(tape.ops[sin_idx], ad::OpCode::SIN);
  const std::size_t sin_off = operand_offset(tape, sin_idx);
  const std::size_t p2 = tape.operands[sin_off];

  const std::size_t expected[3] = {x.tape_index(), y.tape_index(),
                                   z.tape_index()};
  EXPECT_NE(std::find(std::begin(expected), std::end(expected), p0),
            std::end(expected));
  EXPECT_NE(std::find(std::begin(expected), std::end(expected), p1),
            std::end(expected));
  EXPECT_NE(std::find(std::begin(expected), std::end(expected), p2),
            std::end(expected));
}
