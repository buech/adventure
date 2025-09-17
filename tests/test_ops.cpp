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
#include "adventure/ops.hpp"
#include <cmath>

using namespace adventure;

TEST(Ops, Coefficients) {
    // Add / Sub
    EXPECT_DOUBLE_EQ(AddOp::coeff_left<double>(2.0, 3.0), 1.0);
    EXPECT_DOUBLE_EQ(AddOp::coeff_right<double>(2.0, 3.0), 1.0);
    EXPECT_DOUBLE_EQ(SubOp::coeff_left<double>(2.0, 3.0), 1.0);
    EXPECT_DOUBLE_EQ(SubOp::coeff_right<double>(2.0, 3.0), -1.0);

    // Mul
    EXPECT_DOUBLE_EQ(MulOp::coeff_left<double>(2.0, 3.0), 3.0);
    EXPECT_DOUBLE_EQ(MulOp::coeff_right<double>(2.0, 3.0), 2.0);

    // Div
    EXPECT_DOUBLE_EQ(DivOp::coeff_left<double>(2.0, 3.0), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(DivOp::coeff_right<double>(2.0, 3.0), -2.0 / (3.0 * 3.0));

    // Unary derivatives
    EXPECT_DOUBLE_EQ(SinOp::derivative<double>(0.5), std::cos(0.5));
    EXPECT_DOUBLE_EQ(CosOp::derivative<double>(0.5), -std::sin(0.5));
    EXPECT_DOUBLE_EQ(ExpOp::derivative<double>(0.5), std::exp(0.5));
    EXPECT_DOUBLE_EQ(LogOp::derivative<double>(2.0), 1.0 / 2.0);
    EXPECT_DOUBLE_EQ(TanhOp::derivative<double>(0.5), 1.0 - std::tanh(0.5) * std::tanh(0.5));
    EXPECT_DOUBLE_EQ(TanOp::derivative<double>(0.5), 1.0 / (std::cos(0.5) * std::cos(0.5)));
    EXPECT_DOUBLE_EQ(AtanOp::derivative<double>(0.5), 1.0 / (1.0 + 0.5 * 0.5));
    EXPECT_DOUBLE_EQ(AbsOp::derivative<double>(-3.0), -1.0);
    EXPECT_DOUBLE_EQ(SqrtOp::derivative<double>(4.0), 0.5 / std::sqrt(4.0));
    EXPECT_DOUBLE_EQ(CbrtOp::derivative<double>(8.0), 1.0 / (3.0 * std::cbrt(8.0) * std::cbrt(8.0)));
    // PowOp derivative coefficients (left-partial and right-partial)
    //   d(a^b)/da = b * a^(b-1)
    //   d(a^b)/db = a^b * log(a)
    EXPECT_DOUBLE_EQ(PowOp::coeff_left<double>(2.0, 3.0),
                     3.0 * std::pow(2.0, 2.0));
    EXPECT_DOUBLE_EQ(PowOp::coeff_right<double>(2.0, 3.0),
                     std::pow(2.0, 3.0) * std::log(2.0));
    // Value helpers - verify that each Op can compute the forward value.
    EXPECT_DOUBLE_EQ(AddOp::value<double>(2.0, 3.0), 5.0);
    EXPECT_DOUBLE_EQ(SubOp::value<double>(5.0, 3.0), 2.0);
    EXPECT_DOUBLE_EQ(MulOp::value<double>(2.0, 3.0), 6.0);
    EXPECT_DOUBLE_EQ(DivOp::value<double>(6.0, 3.0), 2.0);
    EXPECT_DOUBLE_EQ(PowOp::value<double>(2.0, 3.0), std::pow(2.0, 3.0));
    EXPECT_DOUBLE_EQ(SinOp::value<double>(0.5), std::sin(0.5));
    EXPECT_DOUBLE_EQ(CosOp::value<double>(0.5), std::cos(0.5));
    EXPECT_DOUBLE_EQ(ExpOp::value<double>(0.5), std::exp(0.5));
    EXPECT_DOUBLE_EQ(LogOp::value<double>(2.0), std::log(2.0));
    // Log10 operation tests
    EXPECT_DOUBLE_EQ(Log10Op::derivative<double>(2.0), 1.0 / (2.0 * std::log(10.0)));
    EXPECT_DOUBLE_EQ(Log10Op::value<double>(0.5), std::log10(0.5));
    EXPECT_DOUBLE_EQ(TanhOp::value<double>(0.5), std::tanh(0.5));
    EXPECT_DOUBLE_EQ(TanOp::value<double>(0.5), std::tan(0.5));
    EXPECT_DOUBLE_EQ(AtanOp::value<double>(0.5), std::atan(0.5));
    EXPECT_DOUBLE_EQ(AbsOp::value<double>(-3.0), std::abs(-3.0));
    EXPECT_DOUBLE_EQ(AbsOp::value<double>( 0.0), std::abs( 0.0));
    EXPECT_DOUBLE_EQ(AbsOp::value<double>( 4.2), std::abs( 4.2));
    EXPECT_DOUBLE_EQ(CbrtOp::value<double>(27.0), std::cbrt(27.0));
    EXPECT_DOUBLE_EQ(SqrtOp::value<double>(4.0), std::sqrt(4.0));

    // Min/Max forward value and derivative coefficients
    EXPECT_DOUBLE_EQ(MinOp::value<double>(2.0, 5.0), 2.0);
    EXPECT_DOUBLE_EQ(MaxOp::value<double>(2.0, 5.0), 5.0);
    EXPECT_DOUBLE_EQ(MinOp::coeff_left<double>(2.0, 5.0), 1.0);
    EXPECT_DOUBLE_EQ(MinOp::coeff_right<double>(2.0, 5.0), 0.0);
    EXPECT_DOUBLE_EQ(MaxOp::coeff_left<double>(2.0, 5.0), 0.0);
    EXPECT_DOUBLE_EQ(MaxOp::coeff_right<double>(2.0, 5.0), 1.0);
}

// Unary minus operator tag tests
TEST(Ops, UnaryMinus) {
    // Value of negation
    EXPECT_DOUBLE_EQ(NegOp::value<double>(3.5), -3.5);
    // Derivative of negation
    EXPECT_DOUBLE_EQ(NegOp::derivative<double>(3.5), -1.0);
}
