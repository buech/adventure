/*
 * Copyright (c) 2025-2026 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#include <gtest/gtest.h>

#include <cmath>

#include "adventure/tape.hpp"
#include "adventure/variable.hpp"

namespace ad = adventure;

#ifdef ADVENTURE_ENABLE_LLVM_JIT

template <typename Scalar>
void expect_near(const Scalar &actual, const Scalar &expected,
                 const Scalar &tol = Scalar(1e-6)) {
  EXPECT_NEAR(actual, expected, tol);
}

TEST(TapeJITTest, BasicArithmeticPrimalAndBackward) {
  auto &tape = ad::get_tape<double>();
  tape.clear();

  ad::Variable<double> x(2.0);
  tape.register_input(x);

  ad::Variable<double> y = x * x + 3.0 * x + 1.0;
  tape.register_output(y);

  // Compile JIT
  ASSERT_TRUE(tape.compile_jit());
  EXPECT_TRUE(tape.is_jit_compiled());

  // Seed gradient: dy/dy = 1
  y.grad() = 1.0;
  tape.backward();  // Runs JIT-compiled backward

  // dy/dx = 2*x + 3 = 2*(2.0) + 3 = 7.0
  expect_near(x.grad(), 7.0);

  // Now change input, re-evaluate primal using JIT, and recompute backward
  tape.reset_adj();
  x.value() = 5.0;
  tape.primal();  // Runs JIT-compiled primal

  // y(5) = 5^2 + 3*5 + 1 = 25 + 15 + 1 = 41.0
  expect_near(y.value(), 41.0);

  y.grad() = 1.0;
  tape.backward();
  // dy/dx at x=5: 2*5 + 3 = 13.0
  expect_near(x.grad(), 13.0);
}

TEST(TapeJITTest, TranscendentalFunctions) {
  auto &tape = ad::get_tape<double>();
  tape.clear();

  ad::Variable<double> x(0.5);
  tape.register_input(x);

  ad::Variable<double> y = sin(x) + cos(x) + exp(x) + log(x) + sqrt(x);
  tape.register_output(y);

  ASSERT_TRUE(tape.compile_jit());

  y.grad() = 1.0;
  tape.backward();

  double expected_grad = std::cos(0.5) - std::sin(0.5) + std::exp(0.5) +
                         (1.0 / 0.5) + (0.5 / std::sqrt(0.5));
  expect_near(x.grad(), expected_grad);

  // Evaluate primal at new point
  tape.reset_adj();
  x.value() = 1.2;
  tape.primal();

  double expected_val = std::sin(1.2) + std::cos(1.2) + std::exp(1.2) +
                        std::log(1.2) + std::sqrt(1.2);
  expect_near(y.value(), expected_val);

  y.grad() = 1.0;
  tape.backward();
  double expected_grad_12 = std::cos(1.2) - std::sin(1.2) + std::exp(1.2) +
                            (1.0 / 1.2) + (0.5 / std::sqrt(1.2));
  expect_near(x.grad(), expected_grad_12);
}

TEST(TapeJITTest, MinMaxPowTanhOps) {
  auto &tape = ad::get_tape<double>();
  tape.clear();

  ad::Variable<double> a(2.0);
  ad::Variable<double> b(3.0);
  tape.register_input(a);
  tape.register_input(b);

  ad::Variable<double> y = min(a, b) + max(a, b) + pow(a, b) + tanh(a);
  tape.register_output(y);

  ASSERT_TRUE(tape.compile_jit());

  y.grad() = 1.0;
  tape.backward();

  // min(2, 3) gives da = 1, db = 0
  // max(2, 3) gives da = 0, db = 1
  // pow(2, 3) gives da = 3 * 2^2 = 12, db = 2^3 * ln(2) = 8 * ln(2)
  // tanh(2) gives da = 1 - tanh(2)^2, db = 0
  double t = std::tanh(2.0);
  double expected_grad_a = 1.0 + 0.0 + 3.0 * std::pow(2.0, 2.0) + (1.0 - t * t);
  double expected_grad_b = 0.0 + 1.0 + std::pow(2.0, 3.0) * std::log(2.0);

  expect_near(a.grad(), expected_grad_a);
  expect_near(b.grad(), expected_grad_b);

  // Now change primal inputs such that the min/max branch flips: a > b
  tape.reset_adj();
  a.value() = 4.0;
  b.value() = 1.0;
  tape.primal();

  // y(4, 1) = min(4, 1) + max(4, 1) + pow(4, 1) + tanh(4) = 1 + 4 + 4 + tanh(4)
  double expected_val = 1.0 + 4.0 + std::pow(4.0, 1.0) + std::tanh(4.0);
  expect_near(y.value(), expected_val);

  y.grad() = 1.0;
  tape.backward();

  // min(4, 1): 4 > 1 => da = 0, db = 1
  // max(4, 1): 4 > 1 => da = 1, db = 0
  // pow(4, 1): da = 1 * 4^0 = 1, db = 4^1 * ln(4) = 4 * ln(4)
  // tanh(4): da = 1 - tanh(4)^2, db = 0
  double t2 = std::tanh(4.0);
  double expected_grad_a_swapped =
      0.0 + 1.0 + 1.0 * std::pow(4.0, 0.0) + (1.0 - t2 * t2);
  double expected_grad_b_swapped =
      1.0 + 0.0 + std::pow(4.0, 1.0) * std::log(4.0);

  expect_near(a.grad(), expected_grad_a_swapped);
  expect_near(b.grad(), expected_grad_b_swapped);
}

TEST(TapeJITTest, FloatPrecision) {
  auto &tape = ad::get_tape<float>();
  tape.clear();

  ad::Variable<float> x(2.0f);
  tape.register_input(x);

  ad::Variable<float> y = x * x * x;
  tape.register_output(y);

  ASSERT_TRUE(tape.compile_jit());

  y.grad() = 1.0f;
  tape.backward();

  expect_near(x.grad(), 12.0f, 1e-4f);

  tape.reset_adj();
  x.value() = 3.0f;
  tape.primal();
  expect_near(y.value(), 27.0f, 1e-4f);

  y.grad() = 1.0f;
  tape.backward();
  expect_near(x.grad(), 27.0f, 1e-4f);
}

#endif  // ADVENTURE_ENABLE_LLVM_JIT
