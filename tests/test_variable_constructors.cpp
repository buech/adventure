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
#include <utility>   // std::move

#include "adventure/variable.hpp"

namespace ad = adventure;

namespace {

using ScalarTypes = ::testing::Types<double, float>;

template <typename T>
class VariableCtorTest : public ::testing::Test {
 public:
  using Scalar = T;
};
TYPED_TEST_SUITE(VariableCtorTest, ScalarTypes);

// Helper utilities
namespace detail {

// Build a simple expression and return the result by value.
// The function is deliberately separate from the test body so that the
// copy-constructor is exercised when the temporary is returned.
template <typename Scalar>
ad::Variable<Scalar> make_quadventureatic(const ad::Variable<Scalar>& x) {
  // f(x) = x * x
  return x * x; // returns a temporary Variable
}

// Same as above but returns a leaf (no tape node) - useful for checking that
// copy-construction of a constant does not allocate a node.
template <typename Scalar>
ad::Variable<Scalar> make_constant(Scalar v) {
  return ad::Variable<Scalar>(v);   // default-constructed (invalid idx)
}

} // namespace detail

TYPED_TEST(VariableCtorTest, DefaultCtorIsInactive) {
  using Scalar = typename TestFixture::Scalar;

  ad::Variable<Scalar> v;
  EXPECT_EQ(v.tape_index(), ad::Variable<Scalar>::invalid_idx);
}

TYPED_TEST(VariableCtorTest, CopyCtorInactive) {
  using Scalar = typename TestFixture::Scalar;

  auto &tape = ad::get_tape<Scalar>();
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> a;
  std::size_t tape_size_before = tape.size();

  ad::Variable<Scalar> b(a);
  EXPECT_EQ(b.tape_index(), ad::Variable<Scalar>::invalid_idx);
  EXPECT_EQ(tape.size(), tape_size_before);
}

TYPED_TEST(VariableCtorTest, CopyCtorLeaf) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(1.0));
  ad::register_input(x);   // x becomes a leaf (idx = 0)

  std::size_t tape_size_before = ad::get_tape<Scalar>().size();

  ad::Variable<Scalar> y(x);           // copy ctor
  EXPECT_EQ(y.tape_index(), x.tape_index()) << "Copy ctor must not allocate a new node";
  EXPECT_EQ(ad::get_tape<Scalar>().size(), tape_size_before);
}

// Copy-construction of a temporary returned from a function/operator.
// The returned object must keep the same index that was recorded inside
// the function.
TYPED_TEST(VariableCtorTest, CopyCtorFromReturnedTemporary) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(2.0));
  ad::register_input(x);

  // Record tape size before creating the temporary.
  std::size_t size_before = ad::get_tape<Scalar>().size();

  // The function creates a temporary (x*x) and returns it.
  ad::Variable<Scalar> y = detail::make_quadventureatic(x);

  // The temporary created inside make_quadventureatic should add exactly ONE new node.
  EXPECT_EQ(ad::get_tape<Scalar>().size(), size_before + 1);
  // y must refer to that node.
  EXPECT_NE(y.tape_index(), ad::Variable<Scalar>::invalid_idx);
}

TYPED_TEST(VariableCtorTest, MoveCtorTransfersNode) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> a(Scalar(5.0));
  ad::register_input(a);   // a is a leaf (idx = 0)

  ad::Variable<Scalar> b(std::move(a)); // move ctor

  // b must now hold the original index.
  EXPECT_NE(b.tape_index(), ad::Variable<Scalar>::invalid_idx);
  // a must be left in the null state.
  EXPECT_EQ(a.tape_index(), ad::Variable<Scalar>::invalid_idx);
}

TYPED_TEST(VariableCtorTest, MoveCtorInactive) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> a;
  ad::Variable<Scalar> b(std::move(a));

  EXPECT_EQ(b.tape_index(), ad::Variable<Scalar>::invalid_idx);
  EXPECT_EQ(a.tape_index(), ad::Variable<Scalar>::invalid_idx);
}

 // Interaction with the backward pass - a copy-constructed leaf must
 // behave exactly like the original (gradients are shared).
TYPED_TEST(VariableCtorTest, CopyCtorSharesAdjoint) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(3.0));
  ad::register_input(x);

  // y is a shallow copy of x.
  ad::Variable<Scalar> y(x);

  // Use only y in a downstream computation.
  ad::Variable<Scalar> f = y * y;   // f = y^2
  ad::register_output(f);
  f.grad() = Scalar(1);
  ad::backward<Scalar>();

  // Because y and x share the same tape node, both must have the same
  // gradient (2 * original value).
  Scalar expected = Scalar(2) * x.value();
  EXPECT_EQ(x.grad(), expected);
  EXPECT_EQ(y.grad(), expected);
}

TYPED_TEST(VariableCtorTest, CopyCtorConstant) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> c = detail::make_constant<Scalar>(Scalar(7.0));
  std::size_t tape_size_before = ad::get_tape<Scalar>().size();

  ad::Variable<Scalar> d(c);
  EXPECT_EQ(d.tape_index(), ad::Variable<Scalar>::invalid_idx);
  EXPECT_EQ(ad::get_tape<Scalar>().size(), tape_size_before);
}

TYPED_TEST(VariableCtorTest, SelfCopyCtorNoEffect) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> v(Scalar(1.0));
  ad::register_input(v);
  std::size_t idx_before = v.tape_index();
  std::size_t tape_before = ad::get_tape<Scalar>().size();

  ad::Variable<Scalar> w(v);
  EXPECT_EQ(w.tape_index(), idx_before);
  EXPECT_EQ(ad::get_tape<Scalar>().size(), tape_before);
}

TYPED_TEST(VariableCtorTest, CopyCtorNonLeaf) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(2.0));
  ad::register_input(x);

  // Build a non-leaf node.
  ad::Variable<Scalar> y = x + Scalar(3.0);
  std::size_t tape_size_before = ad::get_tape<Scalar>().size();

  // Copy-construct y.
  ad::Variable<Scalar> z(y);
  EXPECT_EQ(z.tape_index(), y.tape_index());
  EXPECT_EQ(ad::get_tape<Scalar>().size(), tape_size_before);
}

TYPED_TEST(VariableCtorTest, MoveCtorNonLeaf) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(2.0));
  ad::register_input(x);

  ad::Variable<Scalar> y = x * Scalar(5.0);
  std::size_t idx_before = y.tape_index();

  ad::Variable<Scalar> z(std::move(y));
  EXPECT_EQ(z.tape_index(), idx_before);
  EXPECT_EQ(y.tape_index(), ad::Variable<Scalar>::invalid_idx);
}

TYPED_TEST(VariableCtorTest, CopyCtorDoesNotGrowTape) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(1.0));
  ad::register_input(x);

  std::size_t size_before = ad::get_tape<Scalar>().size();

  ad::Variable<Scalar> a(x);
  ad::Variable<Scalar> b(a);
  [[maybe_unused]] ad::Variable<Scalar> c(b);

  EXPECT_EQ(ad::get_tape<Scalar>().size(), size_before);
}

TYPED_TEST(VariableCtorTest, MoveCtorDoesNotGrowTape) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(1.0));
  ad::register_input(x);

  std::size_t size_before = ad::get_tape<Scalar>().size();

  ad::Variable<Scalar> a(std::move(x));
  EXPECT_EQ(ad::get_tape<Scalar>().size(), size_before);
}

} // namespace
