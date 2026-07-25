/*
 * Copyright (c) 2025 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
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
