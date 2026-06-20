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

#include <utility>

#include "adventure/variable.hpp"

namespace ad = adventure;

namespace {

using ScalarTypes = ::testing::Types<double, float>;

template <typename T>
class AssignmentTest : public ::testing::Test {
 public:
  using Scalar = T;
};

TYPED_TEST_SUITE(AssignmentTest, ScalarTypes);

TYPED_TEST(AssignmentTest, CopyActiveVariable) {
  using Scalar = typename TestFixture::Scalar;

  // Build a tape with a single input variable `x`.
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(3.0));
  ad::register_input(x);  // x is now a leaf node

  // Copy-assign to a second variable `y`.
  ad::Variable<Scalar> y;  // default-constructed (invalid)
  y = x;                   // copy assignment

#ifndef ADVENTURE_SHALLOW_COPY
  EXPECT_NE(x.tape_index(), y.tape_index())
      << "Copy should allocate a new node";
#else
  EXPECT_EQ(x.tape_index(), y.tape_index())
      << "Copy should not allocate a new node";
#endif

  EXPECT_EQ(x.value(), y.value());

  // Use only `y` in a downstream computation: f(y) = y * y
  ad::Variable<Scalar> f = y * y;
  ad::register_output(f);
  f.grad() = Scalar(1);
  ad::backward<Scalar>();

  // Gradient of `y` must be the correct derivative 2*y.
  Scalar expected_y_grad = Scalar(2) * y.value();
  // Gradient of `x` must receive the same contribution because `y` depends on
  // `x`.
  EXPECT_EQ(x.grad(), expected_y_grad);
  EXPECT_EQ(y.grad(), expected_y_grad);
}

TYPED_TEST(AssignmentTest, MoveActiveVariableTransfersNode) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> a(Scalar(5.0));
  ad::register_input(a);  // a is a leaf node

  // Move-assign into `b`.
  ad::Variable<Scalar> b = std::move(a);

  // After the move, `b` must hold a valid index and `a` must be invalid.
  EXPECT_NE(b.tape_index(), ad::Variable<Scalar>::invalid_idx);
  EXPECT_EQ(a.tape_index(), ad::Variable<Scalar>::invalid_idx)
      << "Source variable should be left in a null state";

  // Use `b` in a simple expression to make sure the node is still usable.
  ad::Variable<Scalar> g = b + Scalar(2);
  ad::register_output(g);
  g.grad() = Scalar(1);
  ad::backward<Scalar>();

  // The derivative of (b + 2) w.r.t. b is 1, so `b`'s gradient must be 1.
  EXPECT_EQ(b.grad(), Scalar(1));
}

TYPED_TEST(AssignmentTest, SelfAssignmentLeavesNodeUnchanged) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> v(Scalar(7.0));
  ad::register_input(v);

  std::size_t old_idx = v.tape_index();
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
  v = v;
#pragma clang diagnostic pop

  EXPECT_EQ(v.tape_index(), old_idx)
      << "Self-assignment must not change the node";
}

TYPED_TEST(AssignmentTest, CompoundAddAssign) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(2.0));
  ad::register_input(x);

  // y = x; then y += 3
  ad::Variable<Scalar> y = x;
  y += Scalar(3.0);
  EXPECT_EQ(y.value(), Scalar(5.0));

  ad::register_output(y);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  EXPECT_EQ(x.grad(), Scalar(1));
}

TYPED_TEST(AssignmentTest, CompoundMulAssign) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(4.0));
  ad::register_input(x);

  // y = x; then y *= 2
  ad::Variable<Scalar> y = x;
  y *= Scalar(2.0);
  EXPECT_EQ(y.value(), Scalar(8.0));

  ad::register_output(y);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  // dy/dx = 2
  EXPECT_EQ(x.grad(), Scalar(2));
}

TYPED_TEST(AssignmentTest, CompoundAddAssignExpr) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(2.0));
  ad::register_input(x);

  // y = x; then y += x * x  (y = x + x^2)
  ad::Variable<Scalar> y = x;
  y += x * x;
  // Expected value: x + x^2 = 2 + 4 = 6
  EXPECT_EQ(y.value(), Scalar(6));

  ad::register_output(y);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  // dy/dx = 1 + 2*x = 1 + 4 = 5
  EXPECT_EQ(x.grad(), Scalar(5));
}

TYPED_TEST(AssignmentTest, CompoundMulAssignExpr) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(2.0));
  ad::register_input(x);

  // y = x; then y *= x + x  (y = x * (2x) = 2x^2)
  ad::Variable<Scalar> y = x;
  y *= x + x;
  // Expected value: 2 * 2^2 = 8
  EXPECT_EQ(y.value(), Scalar(8));

  ad::register_output(y);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  // dy/dx = 4*x = 8
  EXPECT_EQ(x.grad(), Scalar(8));
}

TYPED_TEST(AssignmentTest, AssignFromPassive) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(3.0));
  ad::register_input(x);

  // Assign a passive scalar value.
  x = Scalar(5.0);

  // After assignment, x should become inactive and hold the new value.
  EXPECT_EQ(x.tape_index(), ad::Variable<Scalar>::invalid_idx);
  EXPECT_EQ(x.value(), Scalar(5.0));

  // Reactivate by registering input again.
  ad::register_input(x);
  // Compute a simple expression to test tape.
  ad::Variable<Scalar> y = x * x;
  ad::register_output(y);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  EXPECT_EQ(x.grad(), Scalar(2) * Scalar(5.0));
}

}  // namespace
