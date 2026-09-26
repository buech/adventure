/*
 * Copyright (c) 2025 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include "adventure/config.hpp"
#include "adventure/expr.hpp"

namespace adventure {

/// Represents a scalar variable participating in forward-mode AD.
/**
 * @tparam T Scalar type (default: double).
 * @tparam N The number of derivative directions (default: 1).
 */
template <typename T = double, std::size_t N = 1>
class Tangent : public ExprBase<Tangent<T, N>, T> {
 public:
  using StorageType = Tangent<T, N>;

  /// Default constructor (uninitialized).
  constexpr ADVENTURE_STRONG_INLINE Tangent() noexcept = default;

  /// Construct a variable with a given value.
  /**
   * @param v Primal value of the variable.
   */
  constexpr ADVENTURE_STRONG_INLINE Tangent(T v) noexcept
      : derivatives_(), primal_(v) {}

  constexpr ADVENTURE_STRONG_INLINE Tangent(
      T v, const std::array<T, N> &derivatives) noexcept
      : derivatives_(derivatives), primal_(v) {}

  /// Copy constructor.
  constexpr ADVENTURE_STRONG_INLINE Tangent(const Tangent &other) noexcept =
      default;

  /// Move constructor
  /**
   * Transfers ownership of the tape index and primal value, leaving the source
   * in an invalid state.
   */
  constexpr ADVENTURE_STRONG_INLINE Tangent(Tangent &&other) noexcept
      : derivatives_(std::move(other.derivatives_)), primal_(other.primal_) {
    other.primal_ = T(0);
  }

  constexpr ~Tangent() noexcept = default;

  constexpr ADVENTURE_STRONG_INLINE const T &value_impl() const noexcept {
    return primal_;
  }

  constexpr ADVENTURE_STRONG_INLINE T &value_impl() noexcept { return primal_; }

  template <class Writer>
  ADVENTURE_STRONG_INLINE void derivative_impl(Writer &&w,
                                               T coeff) const noexcept {
    w(derivatives_, coeff);
  }

  /// Retrieve (or set) the derivatives of this variable.
  auto &grad() { return derivatives_; }

  const auto &grad() const { return derivatives_; }

  /// Copy-assignment.
  ADVENTURE_STRONG_INLINE Tangent &operator=(const Tangent &other) noexcept =
      default;

  /// Move-assignment.
  ADVENTURE_STRONG_INLINE Tangent &operator=(Tangent &&other) noexcept {
    if (this != &other) {
      derivatives_ = std::move(other.derivatives_);
      primal_ = other.primal_;
      other.primal_ = T(0);
    }
    return *this;
  }

  /// Assignment from a passive scalar value.
  /**
   * The variable becomes inactive (the derivatives are set to zero).
   */
  ADVENTURE_STRONG_INLINE Tangent &operator=(T rhs) noexcept {
    std::ranges::fill(derivatives_, T(0));
    primal_ = rhs;
    return *this;
  }

 private:
  /// The derivative storage of this variable.
  std::array<T, N> derivatives_;
  /// Primal value of this variable.
  T primal_;

  template <std::size_t NDeriv, class Expr>
  friend Tangent<typename Expr::scalar_type, NDeriv> materialise_tangent(
      const Expr &e) noexcept;

 public:
  template <class Expr, class = std::enable_if_t<
                            std::is_base_of_v<ExprBase<Expr, T>, Expr>>>
  ADVENTURE_STRONG_INLINE Tangent(const Expr &e)
      : Tangent(materialise_tangent<N>(e)) {}

  template <class Expr, class = std::enable_if_t<
                            std::is_base_of_v<ExprBase<Expr, T>, Expr>>>
  ADVENTURE_STRONG_INLINE Tangent &operator=(const Expr &e) {
    *this = materialise_tangent<N>(e);
    return *this;
  }
};

template <class T, std::size_t N>
struct TangentWriter {
  std::array<T, N> &derivatives;

  explicit TangentWriter(std::array<T, N> &d) noexcept : derivatives(d) {}

  ADVENTURE_STRONG_INLINE void operator()(
      const std::array<T, N> &other_derivatives, T coeff) noexcept {
    if (coeff == T(0)) return;
    for (std::size_t i = 0; i < N; ++i)
      derivatives[i] += coeff * other_derivatives[i];
  }
};

template <class T, std::size_t N>
struct parent_count<Tangent<T, N>> : std::integral_constant<std::size_t, 1> {};

template <std::size_t N, class Expr>
ADVENTURE_STRONG_INLINE Tangent<typename Expr::scalar_type, N>
materialise_tangent(const Expr &e) noexcept {
  using T = typename Expr::scalar_type;
  T primal = e.value();

  // How many distinct parents *could* this expression possibly have?
  constexpr std::size_t MAX_PARENTS = parent_count<Expr>::value;

  // pure constant -> no tape node
  if constexpr (MAX_PARENTS == 0) {
    return Tangent<T, N>(primal);
  }

  Tangent<T, N> result(primal);
  TangentWriter<T, N> writer(result.grad());
  e.derivative(writer, T(1));

  return result;
}

template <class T>
ADVENTURE_STRONG_INLINE auto &operator+=(Tangent<T> &lhs, T rhs) {
  lhs = lhs + rhs;
  return lhs;
}

template <class T>
ADVENTURE_STRONG_INLINE auto &operator-=(Tangent<T> &lhs, T rhs) {
  lhs = lhs - rhs;
  return lhs;
}

template <class T>
ADVENTURE_STRONG_INLINE auto &operator*=(Tangent<T> &lhs, T rhs) {
  lhs = lhs * rhs;
  return lhs;
}

template <class T>
ADVENTURE_STRONG_INLINE auto &operator/=(Tangent<T> &lhs, T rhs) {
  lhs = lhs / rhs;
  return lhs;
}

template <class T, class Expr,
          std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>, int> = 0>
ADVENTURE_STRONG_INLINE auto &operator+=(Tangent<T> &lhs, const Expr &rhs) {
  lhs = lhs + rhs;
  return lhs;
}

template <class T, class Expr,
          std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>, int> = 0>
ADVENTURE_STRONG_INLINE auto &operator-=(Tangent<T> &lhs, const Expr &rhs) {
  lhs = lhs - rhs;
  return lhs;
}

template <class T, class Expr,
          std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>, int> = 0>
ADVENTURE_STRONG_INLINE auto &operator*=(Tangent<T> &lhs, const Expr &rhs) {
  lhs = lhs * rhs;
  return lhs;
}

template <class T, class Expr,
          std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>, int> = 0>
ADVENTURE_STRONG_INLINE auto &operator/=(Tangent<T> &lhs, const Expr &rhs) {
  lhs = lhs / rhs;
  return lhs;
}

}  // namespace adventure
