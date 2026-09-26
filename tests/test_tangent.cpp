/*
 * Copyright (c) 2025 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#include <gtest/gtest.h>

#include <cmath>

#include "adventure/tangent.hpp"

namespace ad = adventure;

TEST(Tangent, BasicAD) {
  using Scalar = double;
  using AD = ad::Tangent<Scalar, 3>;

  Scalar vx{3.0};
  Scalar vy{2.5};
  Scalar vz{5.3};

  Scalar dx{1.2};
  Scalar dy{2.3};
  Scalar dz{3.4};

  AD x(vx, {dx, 0, 0});
  AD y(vy, {0, dy, 0});
  AD z(vz, {0, 0, dz});

  AD r = 2 * z * sin(x * y);

  EXPECT_DOUBLE_EQ(r.grad()[0], dx * 2 * vz * std::cos(vx * vy) * vy);
  EXPECT_DOUBLE_EQ(r.grad()[1], dy * 2 * vz * std::cos(vx * vy) * vx);
  EXPECT_DOUBLE_EQ(r.grad()[2], dz * 2 * std::sin(vx * vy));
}
