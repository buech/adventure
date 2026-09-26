/*
 * Copyright (c) 2025 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#include "adventure/expr.hpp"
#include "adventure/variable.hpp"

namespace ad = adventure;

static_assert(ad::parent_count<ad::ConstExpr<double>>::value == 0,
              "ConstExpr should have 0 parents");
static_assert(ad::parent_count<ad::Variable<double>>::value == 1,
              "Variable should have 1 parent");
static_assert(
    ad::parent_count<ad::BinExpr<ad::AddOp, ad::Variable<double>,
                                 ad::ConstExpr<double>, double>>::value == 1,
    "Binary node with one leaf and one const should have 1 parent");
static_assert(ad::parent_count<ad::UnaryExpr<ad::SinOp, ad::Variable<double>,
                                             double>>::value == 1,
              "SinExpr of a variable should have 1 parent");
static_assert(
    ad::parent_count<ad::UnaryExpr<ad::LogOp,
                                   ad::BinExpr<ad::MulOp, ad::Variable<double>,
                                               ad::Variable<double>, double>,
                                   double>>::value == 2,
    "LogExpr of a binary product should have 2 parents");
