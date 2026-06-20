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
