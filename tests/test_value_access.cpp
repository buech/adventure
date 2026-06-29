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

namespace {

using ScalarTypes = ::testing::Types<double, float>;

template <typename T>
class ValueAccessTest : public ::testing::Test {
 public:
  using Scalar = T;
};
TYPED_TEST_SUITE(ValueAccessTest, ScalarTypes);

TYPED_TEST(ValueAccessTest, ConstValueAccess) {
  using Scalar = typename TestFixture::Scalar;
  const ad::Variable<Scalar> v(Scalar(3.14));

  EXPECT_EQ(v.value(), Scalar(3.14));
  // Verify it's the same as value_impl()
  EXPECT_EQ(v.value(), v.value_impl());
}

TYPED_TEST(ValueAccessTest, NonConstValueAccessAndMutation) {
  using Scalar = typename TestFixture::Scalar;
  ad::Variable<Scalar> v(Scalar(1.0));

  EXPECT_EQ(v.value(), Scalar(1.0));

  // Mutate via value()
  v.value() = Scalar(2.5);
  EXPECT_EQ(v.value(), Scalar(2.5));
  EXPECT_EQ(v.value_impl(), Scalar(2.5));
}

TYPED_TEST(ValueAccessTest, ExprBaseValueAccess) {
  using Scalar = typename TestFixture::Scalar;
  ad::Variable<Scalar> v(Scalar(5.0));
  const ad::ExprBase<ad::Variable<Scalar>, Scalar> &e = v;

  // This calls ExprBase::value() which calls Variable::value_impl()
  EXPECT_EQ(e.value(), Scalar(5.0));
}

TYPED_TEST(ValueAccessTest, MutationDoesNotAffectTapeActivity) {
  using Scalar = typename TestFixture::Scalar;

  auto &tape = ad::get_tape<Scalar>();
  tape.clear();

  ad::Variable<Scalar> v(Scalar(1.0));
  EXPECT_FALSE(v.is_active());

  v.value() = Scalar(2.0);
  EXPECT_FALSE(v.is_active());
  EXPECT_EQ(v.value(), Scalar(2.0));

  tape.register_input(v);
  EXPECT_TRUE(v.is_active());

  v.value() = Scalar(3.0);
  EXPECT_TRUE(v.is_active());
  EXPECT_EQ(v.value(), Scalar(3.0));
}

TYPED_TEST(ValueAccessTest, ValueReturnType) {
  using Scalar = typename TestFixture::Scalar;
  ad::Variable<Scalar> v(Scalar(1.0));
  ad::Variable<Scalar> w(Scalar(2.0));
  auto expr = v + w;

  // Variable::value() should return a reference (lvalue)
  static_assert(std::is_lvalue_reference_v<decltype(v.value())>,
                "Variable::value() must return an lvalue reference");

  // BinExpr::value() should return by value (prvalue)
  static_assert(!std::is_lvalue_reference_v<decltype(expr.value())>,
                "BinExpr::value() must return by value");
}

}  // namespace
