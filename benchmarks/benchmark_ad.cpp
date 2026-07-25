/*
 * Copyright (c) 2025 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#include <benchmark/benchmark.h>

#include <cmath>

#include "adventure/variable.hpp"

namespace ad = adventure;

template <typename Scalar>
static void BM_Gradient(benchmark::State &state) {
  for (auto _ : state) {
    ad::clear_tape<Scalar>();
    ad::Variable<Scalar> x(Scalar(1.0));
    ad::register_input(x);
    ad::Variable<Scalar> y = x * x * x + exp(x) + log(x) / Scalar(2.0) * x +
                             Scalar(2.0) * (x + Scalar(1.0));
    ad::register_output(y);
    y.grad() = Scalar(1);
    ad::backward<Scalar>();
    benchmark::DoNotOptimize(x.grad());
  }
  state.counters["tape_nodes"] =
      static_cast<double>(ad::get_tape<Scalar>().size());
}

// Register benchmarks for the required scalar types.
BENCHMARK_TEMPLATE(BM_Gradient, double);
BENCHMARK_TEMPLATE(BM_Gradient, float);

BENCHMARK_MAIN();
