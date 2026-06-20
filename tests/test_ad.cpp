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

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "adventure/variable.hpp"

namespace ad = adventure;

// Helper to compute the gradient of a unary function for a given scalar type.
template <typename Scalar, typename Func>
Scalar unary_gradient(Scalar x, Func f) {
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> var(x);
  ad::register_input(var);
  ad::Variable<Scalar> y = f(var);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  return var.grad();
}

/// Helper for binary functions that return a single output variable.
/**
 * @return The gradients w.r.t. the two inputs.
 */
template <typename Scalar, typename Func>
std::pair<Scalar, Scalar> binary_gradient(Scalar x, Scalar y, Func f) {
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> vx(x);
  ad::Variable<Scalar> vy(y);
  ad::register_input(vx);
  ad::register_input(vy);
  ad::Variable<Scalar> out = f(vx, vy);
  out.grad() = Scalar(1);
  ad::backward<Scalar>();
  return {vx.grad(), vy.grad()};
}

/// Compare two scalar values with a tolerance.
template <typename Scalar>
void expect_near(const Scalar &actual, const Scalar &expected) {
  if constexpr (std::is_same_v<Scalar, double>) {
    EXPECT_DOUBLE_EQ(actual, expected);
  } else if constexpr (std::is_same_v<Scalar, float>) {
    EXPECT_FLOAT_EQ(actual, expected);
  }
}

using ScalarTypes = ::testing::Types<double, float>;

template <typename T>
class ADTest : public ::testing::Test {
 public:
  using Scalar = T;
};

TYPED_TEST_SUITE(ADTest, ScalarTypes);

/// Verify that the tape is cleared between tests.
TYPED_TEST(ADTest, TapeClear) {
  using Scalar = typename TestFixture::Scalar;
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(Scalar(1));
  ad::register_input(x);
  ad::Variable<Scalar> y = x + x;
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar grad1 = x.grad();

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x2(Scalar(1));
  ad::register_input(x2);
  ad::Variable<Scalar> y2 = x2 + x2;
  y2.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar grad2 = x2.grad();

  expect_near(grad1, grad2);
}

/// Simple gradient test.
TYPED_TEST(ADTest, SimpleGradient) {
  using Scalar = typename TestFixture::Scalar;
  ad::clear_tape<Scalar>();
  Scalar x_val = Scalar(2);
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = x * x + sin(x);
  expect_near(y.value(), x_val * x_val + std::sin(x_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar expected = Scalar(2) * Scalar(2) + std::cos(x_val);
  expect_near(x.grad(), expected);
}

/// Gradient of sin(x) should be cos(x).
TYPED_TEST(ADTest, SinGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(0.5);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> var(x_val);
  ad::register_input(var);
  ad::Variable<Scalar> y = sin(var);
  expect_near(y.value(), std::sin(x_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar grad = var.grad();
  expect_near(grad, std::cos(x_val));
}

/// Gradient of cos(x) should be -sin(x).
TYPED_TEST(ADTest, CosGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(0.7);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> var(x_val);
  ad::register_input(var);
  ad::Variable<Scalar> y = cos(var);
  expect_near(y.value(), std::cos(x_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar grad = var.grad();
  expect_near(grad, -std::sin(x_val));
}

/// Gradient of exp(x) should be exp(x).
TYPED_TEST(ADTest, ExpGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(1.2);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> var(x_val);
  ad::register_input(var);
  ad::Variable<Scalar> y = exp(var);
  expect_near(y.value(), std::exp(x_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar grad = var.grad();
  expect_near(grad, std::exp(x_val));
}

/// Gradient of log(x) should be 1/x.
TYPED_TEST(ADTest, LogGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(3.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> var(x_val);
  ad::register_input(var);
  ad::Variable<Scalar> y = log(var);
  expect_near(y.value(), std::log(x_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar grad = var.grad();
  expect_near(grad, Scalar(1) / x_val);
}

// Gradient test for log10(x)
TYPED_TEST(ADTest, Log10Gradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(3.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> var(x_val);
  ad::register_input(var);
  ad::Variable<Scalar> y = log10(var);
  expect_near(y.value(), std::log10(x_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar expected = Scalar(1) / (x_val * std::log(Scalar(10)));
  expect_near(var.grad(), expected);
}

TYPED_TEST(ADTest, TanhGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(0.9);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> var(x_val);
  ad::register_input(var);
  ad::Variable<Scalar> y = tanh(var);
  expect_near(y.value(), std::tanh(x_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar grad = var.grad();
  expect_near(grad, Scalar(1) - std::tanh(x_val) * std::tanh(x_val));
}

TYPED_TEST(ADTest, SqrtGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(4.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> var(x_val);
  ad::register_input(var);
  ad::Variable<Scalar> y = sqrt(var);
  expect_near(y.value(), std::sqrt(x_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar expected = Scalar(0.5) / std::sqrt(x_val);
  expect_near(var.grad(), expected);
}

TYPED_TEST(ADTest, CbrtGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(8.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> var(x_val);
  ad::register_input(var);
  ad::Variable<Scalar> y = cbrt(var);
  expect_near(y.value(), std::cbrt(x_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar expected =
      Scalar(1) / (Scalar(3) * std::cbrt(x_val) * std::cbrt(x_val));
  expect_near(var.grad(), expected);
}

TYPED_TEST(ADTest, AbsGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(-3.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> var(x_val);
  ad::register_input(var);
  ad::Variable<Scalar> y = abs(var);
  expect_near(y.value(), std::abs(x_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  // derivative: sign(x)
  Scalar expected = (x_val > Scalar(0))
                        ? Scalar(1)
                        : ((x_val < Scalar(0)) ? Scalar(-1) : Scalar(0));
  expect_near(var.grad(), expected);
}

TYPED_TEST(ADTest, AbsGradientVarious) {
  using Scalar = typename TestFixture::Scalar;

  // Helper lambda to run a single case.
  auto run = [&](Scalar x_val, Scalar expected_grad) {
    ad::clear_tape<Scalar>();
    ad::Variable<Scalar> var(x_val);
    ad::register_input(var);
    ad::Variable<Scalar> y = abs(var);
    expect_near(y.value(), std::abs(x_val));
    ad::register_output(y);
    y.grad() = Scalar(1);
    ad::backward<Scalar>();
    expect_near(var.grad(), expected_grad);
  };

  // negative input -> gradient = -1
  run(Scalar(-3.0), Scalar(-1));

  // positive input -> gradient = +1
  run(Scalar(2.5), Scalar(1));

  // zero input -> gradient = 0 (we returns 0 at x==0)
  run(Scalar(0.0), Scalar(0));
}

TYPED_TEST(ADTest, MinGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar a_val = Scalar(2.0);
  Scalar b_val = Scalar(5.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> a(a_val);
  ad::Variable<Scalar> b(b_val);
  ad::register_input(a);
  ad::register_input(b);
  ad::Variable<Scalar> y = min(a, b);
  expect_near(y.value(), std::min(a_val, b_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  if (a_val < b_val) {
    expect_near(a.grad(), Scalar(1));
    expect_near(b.grad(), Scalar(0));
  } else {
    expect_near(a.grad(), Scalar(0));
    expect_near(b.grad(), Scalar(1));
  }
}
TYPED_TEST(ADTest, MaxGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar a_val = Scalar(7.0);
  Scalar b_val = Scalar(3.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> a(a_val);
  ad::Variable<Scalar> b(b_val);
  ad::register_input(a);
  ad::register_input(b);
  ad::Variable<Scalar> y = max(a, b);
  expect_near(y.value(), std::max(a_val, b_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  if (a_val > b_val) {
    expect_near(a.grad(), Scalar(1));
    expect_near(b.grad(), Scalar(0));
  } else {
    expect_near(a.grad(), Scalar(0));
    expect_near(b.grad(), Scalar(1));
  }
}

TYPED_TEST(ADTest, ClampGradientInside) {
  using Scalar = typename TestFixture::Scalar;
  // Choose a value strictly inside the interval.
  Scalar x_val = Scalar(3.0);
  Scalar lo_val = Scalar(1.0);
  Scalar hi_val = Scalar(5.0);

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::Variable<Scalar> lo(lo_val);
  ad::Variable<Scalar> hi(hi_val);
  ad::register_input(x);
  ad::register_input(lo);
  ad::register_input(hi);

  // y = clamp(x, lo, hi) -> should be x because x is inside
  ad::Variable<Scalar> y = clamp(x, lo, hi);
  expect_near(y.value(), x_val);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();

  // Derivative w.r.t. x is 1, lo/hi get 0.
  expect_near(x.grad(), Scalar(1));
  expect_near(lo.grad(), Scalar(0));
  expect_near(hi.grad(), Scalar(0));
}

TYPED_TEST(ADTest, ClampGradientOutside) {
  using Scalar = typename TestFixture::Scalar;
  // x below the lower bound -> result = lo
  Scalar x_val = Scalar(0.5);
  Scalar lo_val = Scalar(1.0);
  Scalar hi_val = Scalar(5.0);

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::Variable<Scalar> lo(lo_val);
  ad::Variable<Scalar> hi(hi_val);
  ad::register_input(x);
  ad::register_input(lo);
  ad::register_input(hi);

  ad::Variable<Scalar> y = clamp(x, lo, hi);
  expect_near(y.value(), lo_val);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();

  // Derivative w.r.t. x is 0, lo gets the gradient.
  expect_near(x.grad(), Scalar(0));
  expect_near(lo.grad(), Scalar(1));
  expect_near(hi.grad(), Scalar(0));
}
TYPED_TEST(ADTest, AddGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(4.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = x + ad::Variable<Scalar>(Scalar(3.0));
  expect_near(y.value(), x_val + Scalar(3));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  expect_near(x.grad(), Scalar(1));
}

TYPED_TEST(ADTest, SubGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(5.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = x - ad::Variable<Scalar>(Scalar(2.0));
  expect_near(y.value(), x_val - Scalar(2));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  expect_near(x.grad(), Scalar(1));
}

TYPED_TEST(ADTest, MulGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(2.0);
  Scalar c = Scalar(4.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = x * ad::Variable<Scalar>(c);
  expect_near(y.value(), x_val * c);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  expect_near(x.grad(), c);
}

TYPED_TEST(ADTest, DivGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(9.0);
  Scalar c = Scalar(3.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = x / ad::Variable<Scalar>(c);
  expect_near(y.value(), x_val / c);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  expect_near(x.grad(), Scalar(1) / c);
}

TYPED_TEST(ADTest, AddScalarRight) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(4.0);
  Scalar c = Scalar(3.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = x + c;
  expect_near(y.value(), x_val + c);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  expect_near(x.grad(), Scalar(1));
}

TYPED_TEST(ADTest, AddScalarLeft) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(4.0);
  Scalar c = Scalar(3.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = c + x;
  expect_near(y.value(), c + x_val);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  expect_near(x.grad(), Scalar(1));
}

TYPED_TEST(ADTest, SubScalarRight) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(5.0);
  Scalar c = Scalar(2.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = x - c;
  expect_near(y.value(), x_val - c);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  expect_near(x.grad(), Scalar(1));
}

TYPED_TEST(ADTest, SubScalarLeft) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(5.0);
  Scalar c = Scalar(2.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = c - x;
  expect_near(y.value(), c - x_val);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  expect_near(x.grad(), -Scalar(1));
}

TYPED_TEST(ADTest, MulScalarRight) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(2.0);
  Scalar c = Scalar(4.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = x * c;
  expect_near(y.value(), x_val * c);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  expect_near(x.grad(), c);
}

TYPED_TEST(ADTest, MulScalarLeft) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(2.0);
  Scalar c = Scalar(4.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = c * x;
  expect_near(y.value(), c * x_val);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  expect_near(x.grad(), c);
}

TYPED_TEST(ADTest, DivScalarRight) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(9.0);
  Scalar c = Scalar(3.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = x / c;
  expect_near(y.value(), x_val / c);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  expect_near(x.grad(), Scalar(1) / c);
}

TYPED_TEST(ADTest, DivScalarLeft) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(3.0);
  Scalar c = Scalar(9.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = c / x;
  expect_near(y.value(), c / x_val);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  // derivative of c/x w.r.t x is -c / x^2
  expect_near(x.grad(), -c / (x_val * x_val));
}

TYPED_TEST(ADTest, ChainRule) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(0.8);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = sin(x) * exp(x);
  expect_near(y.value(), std::sin(x_val) * std::exp(x_val));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar expected =
      std::cos(x_val) * std::exp(x_val) + std::sin(x_val) * std::exp(x_val);
  expect_near(x.grad(), expected);
}

TYPED_TEST(ADTest, SeededSingleOutput) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(3.0);
  Scalar seed = Scalar(2.5);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = x * x;
  expect_near(y.value(), x_val * x_val);
  y.grad() = seed;
  ad::backward<Scalar>();
  Scalar expected = Scalar(2) * x_val * seed;
  expect_near(x.grad(), expected);
}

// Multi-output backward propagation.
// Two outputs: o1 = a * b, o2 = a + b.
TYPED_TEST(ADTest, MultiOutputGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar a_val = Scalar(2.0);
  Scalar b_val = Scalar(5.0);
  Scalar seed1 = Scalar(1.0);  // seed for o1 = a * b
  Scalar seed2 = Scalar(2.0);  // seed for o2 = a + b

  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> a(a_val);
  ad::Variable<Scalar> b(b_val);
  ad::register_input(a);
  ad::register_input(b);

  // o1 = a * b
  ad::Variable<Scalar> o1 = a * b;
  expect_near(o1.value(), a_val * b_val);
  // o2 = a + b
  ad::Variable<Scalar> o2 = a + b;
  expect_near(o2.value(), a_val + b_val);

  // Set seeds on the output variables
  o1.grad() = seed1;
  o2.grad() = seed2;

  ad::backward<Scalar>();

  // Expected gradients:
  //   do1/da = b, do1/db = a
  //   do2/da = 1, do2/db = 1
  //   grad_a = seed1 * b + seed2 * 1
  //   grad_b = seed1 * a + seed2 * 1
  Scalar expected_a = seed1 * b_val + seed2 * Scalar(1);
  Scalar expected_b = seed1 * a_val + seed2 * Scalar(1);

  expect_near(a.grad(), expected_a);
  expect_near(b.grad(), expected_b);
}

/// Test multiple input variables in a single expression.
TYPED_TEST(ADTest, MultipleInputs) {
  using Scalar = typename TestFixture::Scalar;
  ad::clear_tape<Scalar>();
  Scalar x_val = Scalar(2.0);
  Scalar y_val = Scalar(3.0);
  ad::Variable<Scalar> x(x_val);
  ad::Variable<Scalar> y(y_val);
  ad::register_input(x);
  ad::register_input(y);
  ad::Variable<Scalar> f = x * y + sin(y);
  expect_near(f.value(), x_val * y_val + std::sin(y_val));
  f.grad() = Scalar(1);
  ad::backward<Scalar>();
  // df/dx = y
  // df/dy = x + cos(y)
  Scalar expected_dx = y_val;
  Scalar expected_dy = x_val + std::cos(y_val);
  expect_near(x.grad(), expected_dx);
  expect_near(y.grad(), expected_dy);
}

/// Test zero seed results in zero gradients.
TYPED_TEST(ADTest, ZeroSeed) {
  using Scalar = typename TestFixture::Scalar;
  ad::clear_tape<Scalar>();
  Scalar x_val = Scalar(5.0);
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = x * x;
  expect_near(y.value(), x_val * x_val);
  y.grad() = Scalar(0);
  ad::backward<Scalar>();
  expect_near(x.grad(), Scalar(0));
}

TYPED_TEST(ADTest, PowConstantExponent) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(2.0);
  Scalar expected = Scalar(3) * std::pow(Scalar(2.0), Scalar(2.0));
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> var(x_val);
  ad::register_input(var);
  ad::Variable<Scalar> y = pow(var, Scalar(3.0));
  expect_near(y.value(), std::pow(x_val, Scalar(3.0)));
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar grad = var.grad();
  expect_near(grad, expected);
}

TYPED_TEST(ADTest, PowVariableExponent) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(2.0);
  Scalar e_val = Scalar(4.0);
  expect_near(std::pow(x_val, e_val), std::pow(x_val, e_val));
  auto grads = binary_gradient<Scalar>(
      x_val, e_val,
      [](const ad::Variable<Scalar> &xv, const ad::Variable<Scalar> &ev) {
        return pow(xv, ev);
      });
  Scalar expected_dx = e_val * std::pow(x_val, e_val - Scalar(1));
  Scalar expected_de = std::pow(x_val, e_val) * std::log(x_val);
  expect_near(grads.first, expected_dx);
  expect_near(grads.second, expected_de);
}

TYPED_TEST(ADTest, ComparisonOperators) {
  using Scalar = typename TestFixture::Scalar;
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> a(Scalar(2.0));
  ad::Variable<Scalar> b(Scalar(3.0));
  ad::register_input(a);
  ad::register_input(b);

  EXPECT_TRUE(a < b);
  EXPECT_FALSE(a > b);
  EXPECT_TRUE(a <= b);
  EXPECT_FALSE(a >= b);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a + b > Scalar(4.0));
}

TYPED_TEST(ADTest, UnaryMinus) {
  using Scalar = typename TestFixture::Scalar;

  ad::clear_tape<Scalar>();
  Scalar x_val = Scalar(4.0);
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);

  ad::Variable<Scalar> y = -x;
  EXPECT_EQ(y.value(), -x_val);

  y.grad() = Scalar(1);
  ad::backward<Scalar>();

  EXPECT_EQ(x.grad(), Scalar(-1));
}

TYPED_TEST(ADTest, AtanGradient) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(0.7);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);
  ad::Variable<Scalar> y = atan(x);
  EXPECT_NEAR(y.value(), std::atan(x_val), 1e-12);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  Scalar expected = Scalar(1) / (Scalar(1) + x_val * x_val);
  EXPECT_NEAR(x.grad(), expected, 1e-12);
}

// Test integer and arithmetic literal handling.
TYPED_TEST(ADTest, IntegerLiteralArithmetic) {
  using Scalar = typename TestFixture::Scalar;

  // Right-hand side integer literal
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x = Scalar(2.0);
  ad::register_input(x);
  ad::Variable<Scalar> y = x - 1;
  EXPECT_NEAR(y.value(), Scalar(1.0), 1e-12);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  EXPECT_NEAR(x.grad(), Scalar(1), 1e-12);

  // Left-hand side integer literal
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x2 = Scalar(2.0);
  ad::register_input(x2);
  ad::Variable<Scalar> z = 1 - x2;
  EXPECT_NEAR(z.value(), Scalar(-1.0), 1e-12);
  z.grad() = Scalar(1);
  ad::backward<Scalar>();
  EXPECT_NEAR(x2.grad(), Scalar(-1), 1e-12);
}

// pow with integer exponent (right-hand side literal)
TYPED_TEST(ADTest, PowIntegerExponent) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(2.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);

  // y = x^20 (20 is an int literal)
  ad::Variable<Scalar> y = pow(x, 20);
  EXPECT_NEAR(y.value(), std::pow(x_val, Scalar(20)), 1e-12);

  y.grad() = Scalar(1);
  ad::backward<Scalar>();

  // derivative = 20 * x^(20-1)
  Scalar expected = Scalar(20) * std::pow(x_val, Scalar(19));
  EXPECT_NEAR(x.grad(), expected, 1e-12);
}

// pow with integer base (left-hand side literal)
TYPED_TEST(ADTest, PowIntegerBase) {
  using Scalar = typename TestFixture::Scalar;
  Scalar x_val = Scalar(3.0);
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(x_val);
  ad::register_input(x);

  // y = 5^x (5 is an int literal)
  ad::Variable<Scalar> y = pow(5, x);
  EXPECT_NEAR(y.value(), std::pow(Scalar(5), x_val), 1e-12);

  y.grad() = Scalar(1);
  ad::backward<Scalar>();

  // derivative = 5^x * ln(5)
  Scalar expected = std::pow(Scalar(5), x_val) * std::log(Scalar(5));
  EXPECT_NEAR(x.grad(), expected, 1e-12);
}
