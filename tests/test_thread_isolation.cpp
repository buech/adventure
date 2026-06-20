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

#include <thread>

#include "adventure/expr.hpp"
#include "adventure/tape.hpp"
#include "adventure/variable.hpp"

namespace ad = adventure;

template <typename Scalar>
void thread_body(Scalar input, Scalar /*expected_grad*/) {
  ad::clear_tape<Scalar>();
  ad::Variable<Scalar> x(input);
  ad::register_input(x);
  ad::Variable<Scalar> y = x * x;  // y = x^2
  ad::register_output(y);
  y.grad() = Scalar(1);
  ad::backward<Scalar>();
  EXPECT_NEAR(x.grad(), Scalar(2) * input, 1e-9);
  // After clear, the tape should contain exactly three nodes (1 leaf + 2 unary
  // nodes)
  EXPECT_EQ(ad::get_tape<Scalar>().size(), 3u);
}

TEST(ThreadIsolation, IndependentTapes) {
  using Scalar = double;
  std::thread t1(thread_body<Scalar>, Scalar(3.0), Scalar(6.0));
  std::thread t2(thread_body<Scalar>, Scalar(5.0), Scalar(10.0));
  t1.join();
  t2.join();
}
