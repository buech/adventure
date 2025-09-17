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

#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "adventure/config.hpp"
#include "adventure/tape.hpp"
#include "adventure/expr.hpp"

namespace adventure {

/// Represents a scalar variable participating in reverse-mode AD.
/**
 * Each Variable holds an index into the thread-local tape. Variables are not
 * automatically added to the tape; they must be explicitly registered as
 * inputs using #adventure::register_input. Output variables can be registered with
 * #adventure::register_output, after which their adjoint (gradient) can be set
 * directly via #grad() before calling #adventure::backward.
 *
 * @tparam T Scalar type (default: double).
 */
template <typename T = double>
class Variable : public ExprBase<Variable<T>, T> {
 public:
  using StorageType = Variable<T>;

  static constexpr index_t invalid_idx = static_cast<index_t>(-1);

  /// Default constructor (uninitialized, not tracked).
  constexpr ADVENTURE_STRONG_INLINE Variable() noexcept : idx(invalid_idx), primal_(T(0)) {}

  /// Construct a variable with a given value.
  /**
   * The variable is *not* automatically added to the tape. Use
   * #adventure::register_input to add it as a leaf input.
   *
   * @param v Primal value of the variable.
   */
  constexpr ADVENTURE_STRONG_INLINE Variable(T v) noexcept : idx(invalid_idx), primal_(v) {}

  /// Copy constructor. Shallow copy of the tape index and primal value.
  /**
   * This is used by expression-template returns so that the resulting Variable refers to the same tape node.
   */
  constexpr ADVENTURE_STRONG_INLINE Variable(const Variable& other) noexcept = default;

  /// Move constructor
  /**
   * Transfers ownership of the tape index and primal value, leaving the source in an invalid state.
   */
  constexpr ADVENTURE_STRONG_INLINE Variable(Variable&& other) noexcept
      : idx(other.idx), primal_(other.primal_) {
    other.idx = invalid_idx;
    other.primal_ = T(0);
  }

  constexpr ~Variable() noexcept = default;

  constexpr void set_value(T new_value) noexcept { primal_ = new_value; }

  constexpr ADVENTURE_STRONG_INLINE T value_impl() const noexcept { return primal_; }

  template<class Writer>
  ADVENTURE_STRONG_INLINE void derivative_impl(Writer&& w) const noexcept {
    if (is_active()) {
      w(idx, T(1)); // dx/dx = 1
    }
  }

  /// Retrieve (or set) the gradient (adjoint) of this variable.
  /**
   * For tracked variables this returns a reference to the stored adjoint,
   * allowing the user to assign a seed before calling #adventure::backward.
   */
  T& grad() {
      if (!is_active())
          throw std::logic_error("Only gradients of active variables can be accessed!");
      return get_tape<T>().adj_at(idx);
  }

  const T& grad() const {
    if (!is_active())
        throw std::logic_error("Only gradients of active variables can be accessed!");
    return get_tape<T>().adj_at(idx);
  }

  /// Return the tape index.
  constexpr index_t tape_index() const noexcept { return idx; }

  /// Whether this variable is tracked on a tape.
  constexpr bool is_active() const noexcept { return idx != invalid_idx; }

  /// Copy-assignment.
  ADVENTURE_STRONG_INLINE Variable& operator=(const Variable& other) noexcept {
    if (this != &other) {
#ifndef ADVENTURE_SHALLOW_COPY
      primal_ = other.primal_;
      if (other.is_active()) {
        if (idx == other.idx) return *this;
        // Record y = x as a unary node with derivative 1.
        idx = tape<T>.add_unary(other.idx, T(1));
      } else {
        idx = invalid_idx;
      }
#else
      idx = other.idx;
      primal_ = other.primal_;
#endif
    }
    return *this;
  }

  /// Move-assignment.
  /**
   * Transfers ownership of the tape index to the target and leaves the source in a null (invalid) state.
   */
  ADVENTURE_STRONG_INLINE Variable& operator=(Variable&& other) noexcept {
    if (this != &other) {
      idx = other.idx;
      primal_ = other.primal_;
      other.idx = invalid_idx;
      other.primal_ = T(0);
    }
    return *this;
  }

  /// Assignment from a passive scalar value.
  /**
   * The variable becomes inactive (no tape holds the given value).
   */
  ADVENTURE_STRONG_INLINE Variable& operator=(T rhs) noexcept {
    primal_ = rhs;
    idx = invalid_idx;
    return *this;
  }

private:
  /// Index of this Variable on the tape. #invalid_idx if it is inactive.
  index_t idx;
  /// Primal value of this Variable.
  T primal_;

  /// Construct a Variable from an existing tape index (used internally for tracked results).
  explicit Variable(index_t index, T primal) : idx(index), primal_(primal) {}

  // Grant registration functions access to private members.
  template <typename U>
  friend void register_input(Variable<U>& var);
  template <typename U>
  friend void register_output(Variable<U>& var);

  template<class Expr>
  friend Variable<typename Expr::scalar_type> materialise(const Expr& e) noexcept;

public:
  template<class Expr,
           class = std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>>>
  ADVENTURE_STRONG_INLINE Variable(const Expr& e) : Variable(materialise(e)) {}

  template<class Expr,
           class = std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>>>
  ADVENTURE_STRONG_INLINE Variable& operator=(const Expr& e) {
    *this = materialise(e);
    return *this;
  }
};

 template<class T>
 ADVENTURE_STRONG_INLINE auto& operator+=(Variable<T>& lhs, T rhs) {
   lhs = lhs + rhs;
   return lhs;
 }

 template<class T>
 ADVENTURE_STRONG_INLINE auto& operator-=(Variable<T>& lhs, T rhs) {
   lhs = lhs - rhs;
   return lhs;
 }

 template<class T>
 ADVENTURE_STRONG_INLINE auto& operator*=(Variable<T>& lhs, T rhs) {
   lhs = lhs * rhs;
   return lhs;
 }

 template<class T>
 ADVENTURE_STRONG_INLINE auto& operator/=(Variable<T>& lhs, T rhs) {
   lhs = lhs / rhs;
   return lhs;
 }

 template<class T, class Expr,
          std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>, int> = 0>
 ADVENTURE_STRONG_INLINE auto& operator+=(Variable<T>& lhs, const Expr& rhs) {
   lhs = lhs + rhs;
   return lhs;
 }

 template<class T, class Expr,
          std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>, int> = 0>
 ADVENTURE_STRONG_INLINE auto& operator-=(Variable<T>& lhs, const Expr& rhs) {
   lhs = lhs - rhs;
   return lhs;
 }

 template<class T, class Expr,
          std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>, int> = 0>
 ADVENTURE_STRONG_INLINE auto& operator*=(Variable<T>& lhs, const Expr& rhs) {
   lhs = lhs * rhs;
   return lhs;
 }

 template<class T, class Expr,
          std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>, int> = 0>
 ADVENTURE_STRONG_INLINE auto& operator/=(Variable<T>& lhs, const Expr& rhs) {
   lhs = lhs / rhs;
   return lhs;
 }

/// Register an input variable on the tape.
template <typename T = double>
inline void register_input(Variable<T>& var) {
  auto &tape = get_tape<T>();
  var.idx = tape.add_leaf();
}

template <typename T = double>
inline void register_output(Variable<T>& var) {
  auto &tape = Tape<T>::get_tape();
  if (!var.is_active())
      var.idx = tape.add_leaf();
  else
      var.idx = tape.add_unary(var.idx, T(1));
}

/// Clear the current thread-local tape for a given scalar type.
template <typename T = double>
inline void clear_tape() {
  get_tape<T>().clear();
}

/// Perform a backward pass using the seeds stored in the variables' adjoints (via `grad()`).
template <typename T = double>
inline void backward() {
  get_tape<T>().backward();
}

}  // namespace adventure
