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

TEST(ExceptionSafety, TapeUnchangedAfterException) {
  using Scalar = double;
  auto &tape = ad::get_tape<Scalar>();
  tape.clear();
  std::size_t size_before = tape.size();

  try {
    // Build a variable, then deliberately throw before any tape operation.
    [[maybe_unused]] ad::Variable<Scalar> x(Scalar(1.0));
    throw std::runtime_error("intentional");
  } catch (const std::exception &) {
    // Swallow the exception.
  }

  // Tape size must be unchanged because the exception occurred before any
  // tape-modifying operation.
  EXPECT_EQ(tape.size(), size_before);
}
