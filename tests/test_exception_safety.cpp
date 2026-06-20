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

#include "adventure/expr.hpp"
#include "adventure/tape.hpp"
#include "adventure/variable.hpp"

namespace ad = adventure;

TEST(ExceptionSafety, TapeUnchangedAfterException) {
  using Scalar = double;
  ad::clear_tape<Scalar>();
  std::size_t size_before = ad::get_tape<Scalar>().size();

  try {
    // Build a variable, then deliberately throw before any tape operation.
    [[maybe_unused]] ad::Variable<Scalar> x(Scalar(1.0));
    throw std::runtime_error("intentional");
  } catch (const std::exception &) {
    // Swallow the exception.
  }

  // Tape size must be unchanged because the exception occurred before any
  // tape-modifying operation.
  EXPECT_EQ(ad::get_tape<Scalar>().size(), size_before);
}
