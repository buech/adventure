/*
 * Copyright (c) 2025 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "adventure/config.hpp"
#include "adventure/expr.hpp"
#include "adventure/tape.hpp"

namespace adventure {

/// Represents a scalar variable participating in reverse-mode AD.
/**
 * Each Variable holds an index into the thread-local tape. Variables are not
 * automatically added to the tape; they must be explicitly registered as
 * inputs using #adventure::register_input. Output variables can be registered
 * with #adventure::register_output, after which their adjoint (gradient) can be
 * set directly via #grad() before calling #adventure::backward.
 *
 * @tparam T Scalar type (default: double).
 */
template <typename T = double>
class Variable : public ExprBase<Variable<T>, T> {
 public:
  using StorageType = Variable<T>;

  static constexpr index_t invalid_idx = static_cast<index_t>(-1);

  /// Default constructor (uninitialized, not tracked).
  constexpr ADVENTURE_STRONG_INLINE Variable() noexcept
      : idx(invalid_idx), primal_(T(0)), tape_ptr(nullptr) {}

  /// Construct a variable with a given value.
  /**
   * The variable is *not* automatically added to the tape. Use
   * #adventure::register_input to add it as a leaf input.
   *
   * @param v Primal value of the variable.
   */
  constexpr ADVENTURE_STRONG_INLINE Variable(T v) noexcept
      : idx(invalid_idx), primal_(v), tape_ptr(nullptr) {}

  /// Copy constructor. Shallow copy of the tape index and primal value.
  /**
   * This is used by expression-template returns so that the resulting Variable
   * refers to the same tape node.
   */
  constexpr ADVENTURE_STRONG_INLINE Variable(const Variable &other) noexcept =
      default;

  /// Move constructor
  /**
   * Transfers ownership of the tape index and primal value, leaving the source
   * in an invalid state.
   */
  constexpr ADVENTURE_STRONG_INLINE Variable(Variable &&other) noexcept
      : idx(other.idx), primal_(other.primal_), tape_ptr(other.tape_ptr) {
    other.idx = invalid_idx;
    other.primal_ = T(0);
    other.tape_ptr = nullptr;
  }

  constexpr ~Variable() noexcept = default;

  constexpr ADVENTURE_STRONG_INLINE const T &value_impl() const noexcept {
    return primal_;
  }

  constexpr ADVENTURE_STRONG_INLINE T &value_impl() noexcept { return primal_; }

  template <class Writer>
  ADVENTURE_STRONG_INLINE void derivative_impl(Writer &&w,
                                               T coeff) const noexcept {
    if (is_active()) {
      w(idx, coeff);
    }
  }

  constexpr Tape<T> *get_tape_ptr() const { return tape_ptr; }

  /// Retrieve (or set) the gradient (adjoint) of this variable.
  /**
   * For tracked variables this returns a reference to the stored adjoint,
   * allowing the user to assign a seed before calling #adventure::backward.
   */
  T &grad() {
    if (!is_active())
      throw std::logic_error(
          "Only gradients of active variables can be accessed!");
    if (!tape_ptr)
      throw std::logic_error("Active variable is not attached to a tape!");
    return tape_ptr->adj_at(idx);
  }

  const T &grad() const {
    if (!is_active())
      throw std::logic_error(
          "Only gradients of active variables can be accessed!");
    if (!tape_ptr)
      throw std::logic_error("Active variable is not attached to a tape!");
    return tape_ptr->adj_at(idx);
  }

  /// Return the tape index.
  constexpr index_t tape_index() const noexcept { return idx; }

  /// Whether this variable is tracked on a tape.
  constexpr bool is_active() const noexcept { return idx != invalid_idx; }

  /// Copy-assignment.
  ADVENTURE_STRONG_INLINE Variable &operator=(const Variable &other) noexcept {
    if (this != &other) {
#ifndef ADVENTURE_SHALLOW_COPY
      primal_ = other.primal_;
      if (other.is_active()) {
        if (idx == other.idx) return *this;
        // Record y = x as a unary node with derivative 1.
        tape_ptr = other.tape_ptr;
        idx = tape_ptr->add_unary(other.idx, T(1));
      } else {
        idx = invalid_idx;
        tape_ptr = nullptr;
      }
#else
      idx = other.idx;
      primal_ = other.primal_;
      tape_ptr = other.tape_ptr;
#endif
    }
    return *this;
  }

  /// Move-assignment.
  /**
   * Transfers ownership of the tape index to the target and leaves the source
   * in a null (invalid) state.
   */
  ADVENTURE_STRONG_INLINE Variable &operator=(Variable &&other) noexcept {
    if (this != &other) {
      idx = other.idx;
      primal_ = other.primal_;
      tape_ptr = other.tape_ptr;
      other.idx = invalid_idx;
      other.primal_ = T(0);
      other.tape_ptr = nullptr;
    }
    return *this;
  }

  /// Assignment from a passive scalar value.
  /**
   * The variable becomes inactive (no tape holds the given value).
   */
  ADVENTURE_STRONG_INLINE Variable &operator=(T rhs) noexcept {
    primal_ = rhs;
    idx = invalid_idx;
    tape_ptr = nullptr;
    return *this;
  }

 private:
  /// Index of this Variable on the tape. #invalid_idx if it is inactive.
  index_t idx;
  /// Primal value of this Variable.
  T primal_;
  /// Pointer to the tape that this variable was registered on.
  Tape<T> *tape_ptr;

  /// Construct a Variable from an existing tape index (used internally forl
  /// tracked results).
  explicit Variable(index_t index, T primal, Tape<T> *tape_ptr)
      : idx(index), primal_(primal), tape_ptr(tape_ptr) {}

  // Grant tape access to private members for registration.
  friend Tape<T>;

  template <class Expr>
  friend Variable<typename Expr::scalar_type> materialise(
      const Expr &e) noexcept;

 public:
  template <class Expr, class = std::enable_if_t<
                            std::is_base_of_v<ExprBase<Expr, T>, Expr>>>
  ADVENTURE_STRONG_INLINE Variable(const Expr &e) : Variable(materialise(e)) {}

  template <class Expr, class = std::enable_if_t<
                            std::is_base_of_v<ExprBase<Expr, T>, Expr>>>
  ADVENTURE_STRONG_INLINE Variable &operator=(const Expr &e) {
    *this = materialise(e);
    return *this;
  }
};

/// Variable is a leaf node with a single parent (itself) when active.
template <class T>
struct parent_count<Variable<T>> : std::integral_constant<std::size_t, 1> {};

template <class T>
struct EdgeWriter {
  Tape<T> &tape;
  std::size_t start;
  std::uint8_t used = 0;

  explicit EdgeWriter(Tape<T> &t) noexcept : tape(t), start(t.edges.size()) {}

  ADVENTURE_STRONG_INLINE void operator()(typename Tape<T>::index_type parent,
                                          T coeff) noexcept {
    if (coeff == T(0)) return;

    for (std::uint8_t i = 0; i < used; ++i) {
      auto &edge = tape.edges[start + i];
      if (edge.parent == parent) {
        // same parent, just add the coefficient
        edge.coeff += coeff;
        return;
      }
    }

    tape.edges.emplace_back(parent, coeff);
    ++used;
  }
};

template <class T>
constexpr Tape<T> *get_tape_ptr(const ConstExpr<T> &e) noexcept {
  return nullptr;
}

template <class Op, class E, class T>
constexpr Tape<T> *get_tape_ptr(const UnaryExpr<Op, E, T> &e) noexcept {
  return get_tape_ptr(e.expr_);
}

template <class Op, class L, class R, class T>
constexpr Tape<T> *get_tape_ptr(const BinExpr<Op, L, R, T> &e) noexcept {
  auto *a = get_tape_ptr(e.lhs_);
  auto *b = get_tape_ptr(e.rhs_);
  assert((!a || !b || a == b) &&
         "Tape pointers of lhs and rhs of binary expression do not match.");
  return a ? a : b;
}

template <class T>
constexpr Tape<T> *get_tape_ptr(const Variable<T> &v) noexcept {
  return v.get_tape_ptr();
}

template <class Expr>
ADVENTURE_STRONG_INLINE Variable<typename Expr::scalar_type> materialise(
    const Expr &e) noexcept {
  using T = typename Expr::scalar_type;
  T primal = e.value();

  // How many distinct parents *could* this expression possibly have?
  constexpr std::size_t MAX_PARENTS = parent_count<Expr>::value;

  // pure constant -> no tape node
  if constexpr (MAX_PARENTS == 0) {
    return Variable<T>(primal);
  }

  auto *tape_ptr = get_tape_ptr(e);
  if (!tape_ptr) return Variable<T>(primal);

  EdgeWriter<T> writer(*tape_ptr);
  e.derivative(writer, T(1));

  if (writer.used == 0) {
    return Variable<T>(primal);
  }

  // node metadata, arity = number of distinct parents
  tape_ptr->arities.push_back(writer.used);
  tape_ptr->adj.push_back(0);

  index_t idx = tape_ptr->arities.size() - 1;
  return Variable<T>(idx, primal, tape_ptr);
}

template <class T>
ADVENTURE_STRONG_INLINE auto &operator+=(Variable<T> &lhs, T rhs) {
  lhs = lhs + rhs;
  return lhs;
}

template <class T>
ADVENTURE_STRONG_INLINE auto &operator-=(Variable<T> &lhs, T rhs) {
  lhs = lhs - rhs;
  return lhs;
}

template <class T>
ADVENTURE_STRONG_INLINE auto &operator*=(Variable<T> &lhs, T rhs) {
  lhs = lhs * rhs;
  return lhs;
}

template <class T>
ADVENTURE_STRONG_INLINE auto &operator/=(Variable<T> &lhs, T rhs) {
  lhs = lhs / rhs;
  return lhs;
}

template <class T, class Expr,
          std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>, int> = 0>
ADVENTURE_STRONG_INLINE auto &operator+=(Variable<T> &lhs, const Expr &rhs) {
  lhs = lhs + rhs;
  return lhs;
}

template <class T, class Expr,
          std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>, int> = 0>
ADVENTURE_STRONG_INLINE auto &operator-=(Variable<T> &lhs, const Expr &rhs) {
  lhs = lhs - rhs;
  return lhs;
}

template <class T, class Expr,
          std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>, int> = 0>
ADVENTURE_STRONG_INLINE auto &operator*=(Variable<T> &lhs, const Expr &rhs) {
  lhs = lhs * rhs;
  return lhs;
}

template <class T, class Expr,
          std::enable_if_t<std::is_base_of_v<ExprBase<Expr, T>, Expr>, int> = 0>
ADVENTURE_STRONG_INLINE auto &operator/=(Variable<T> &lhs, const Expr &rhs) {
  lhs = lhs / rhs;
  return lhs;
}

}  // namespace adventure
