#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "adventure/tape.hpp"
#include "adventure/variable.hpp"

namespace ad = adventure;

template <class Var>
void f(const Var &x, Var &y) {
  y = sin(x);
}

TEST(MultiTape, SimpleGradient) {
  ad::Tape<double> tape;
  ad::Variable<double> x(2.0);
  ad::index_t idx_x = tape.register_input(x);
  ad::Variable<double> y;
  f(x, y);
  ad::index_t idx_y = tape.register_output(y);
  tape.set_adj(idx_y, 1.0);
  tape.backward();
  double grad = tape.get_adj(idx_x);
  EXPECT_NEAR(grad, std::cos(2.0), 1e-12);
}

TEST(MultiTape, MultipleTapes) {
  const std::size_t N = 3;
  std::vector<ad::Tape<double>> tapes(N);
  std::vector<ad::index_t> idx_x(N), idx_y(N);
  std::vector<double> inputs = {0.2, 0.4, 0.6};
  std::vector<double> seeds = {1.0, 2.0, 3.0};

  for (std::size_t i = 0; i < N; ++i) {
    auto &tape = tapes[i];
    ad::Variable<double> x(inputs[i]);
    idx_x[i] = tape.register_input(x);
    ad::Variable<double> y;
    f(x, y);
    idx_y[i] = tape.register_output(y);
    tapes[i].set_adj(idx_y[i], seeds[i]);
    tape.backward();
  }

  for (std::size_t i = 0; i < N; ++i) {
    double grad = tapes[i].get_adj(idx_x[i]);
    EXPECT_NEAR(grad, seeds[i] * std::cos(inputs[i]), 1e-12);
  }
}

TEST(MultiTape, ResetAndReevaluate) {
  const std::size_t N = 3;
  std::vector<ad::Tape<double>> tapes(N);
  std::vector<ad::index_t> idx_x(N), idx_y(N);
  std::vector<double> inputs = {0.2, 0.4, 0.6};
  std::vector<double> seeds1 = {1.0, 2.0, 3.0};
  std::vector<double> seeds2 = {2.5, 4.5, 6.5};

  // Record each tape with initial seed
  for (std::size_t i = 0; i < N; ++i) {
    auto &tape = tapes[i];
    ad::Variable<double> x(inputs[i]);
    idx_x[i] = tape.register_input(x);
    ad::Variable<double> y;
    f(x, y);
    idx_y[i] = tape.register_output(y);
    tape.set_adj(idx_y[i], seeds1[i]);
    tape.backward();
    double grad1 = tapes[i].get_adj(idx_x[i]);
    EXPECT_NEAR(grad1, seeds1[i] * std::cos(inputs[i]), 1e-12);
  }

  // Reset adjoints and evaluate with new seeds
  for (std::size_t i = 0; i < N; ++i) {
    auto &tape = tapes[i];
    tape.reset_adj();
    tape.set_adj(idx_y[i], seeds2[i]);
    tape.backward();
    double grad2 = tape.get_adj(idx_x[i]);
    EXPECT_NEAR(grad2, seeds2[i] * std::cos(inputs[i]), 1e-12);
  }
}
