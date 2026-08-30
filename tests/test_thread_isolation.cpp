/*
 * Copyright (c) 2025 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#include <gtest/gtest.h>

#include <thread>

#include "adventure/expr.hpp"
#include "adventure/tape.hpp"
#include "adventure/variable.hpp"

namespace ad = adventure;

template <typename Scalar>
void thread_body(Scalar input, Scalar /*expected_grad*/) {
  auto &tape = ad::get_tape<Scalar>();
  tape.clear();
  ad::Variable<Scalar> x(input);
  tape.register_input(x);
  ad::Variable<Scalar> y = x * x;  // y = x^2
  tape.register_output(y);
  y.grad() = Scalar(1);
  tape.backward();
  EXPECT_NEAR(x.grad(), Scalar(2) * input, 1e-9);
  // After clear, the tape should contain exactly three nodes (1 leaf + 2 unary
  // nodes)
  EXPECT_EQ(tape.size(), 3u);
}

TEST(ThreadIsolation, IndependentTapes) {
  using Scalar = double;
  std::thread t1(thread_body<Scalar>, Scalar(3.0), Scalar(6.0));
  std::thread t2(thread_body<Scalar>, Scalar(5.0), Scalar(10.0));
  t1.join();
  t2.join();
}
