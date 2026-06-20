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

#include <type_traits>

#include "adventure/expr.hpp"

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
