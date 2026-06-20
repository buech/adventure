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
#include <cstdint>
#include <type_traits>
#include <utility>

#include "adventure/config.hpp"
#include "adventure/ops.hpp"
#include "adventure/tape.hpp"

namespace adventure {

// Forward declaration of Variable
template <class T>
class Variable;

// Forward declarations of expression node types
template <class T>
struct ConstExpr;
template <class Op, class L, class R, class T>
struct BinExpr;
template <class Op, class E, class T>
struct UnaryExpr;

template <class Derived, class T>
struct ExprBase {
  using scalar_type = T;
  constexpr ADVENTURE_STRONG_INLINE T value() const {
    return static_cast<const Derived &>(*this).value_impl();
  }

  template <class Writer>
  constexpr ADVENTURE_STRONG_INLINE void derivative(Writer &&w) const {
    static_cast<const Derived &>(*this).derivative_impl(
        std::forward<Writer>(w));
  }
};

template <class E>
struct parent_count;  // primary template (undefined)

template <class T>
struct parent_count<ConstExpr<T>> : std::integral_constant<std::size_t, 0> {};

/// Variable is a leaf node with a single parent (itself) when active.
template <class T>
struct parent_count<Variable<T>> : std::integral_constant<std::size_t, 1> {};

template <class Op, class L, class R, class T>
struct parent_count<BinExpr<Op, L, R, T>>
    : std::integral_constant<std::size_t, parent_count<L>::value +
                                              parent_count<R>::value> {};

template <class Op, class E, class T>
struct parent_count<UnaryExpr<Op, E, T>> : parent_count<E> {};

template <class T>
struct ConstExpr : ExprBase<ConstExpr<T>, T> {
  using StorageType = ConstExpr<T>;

  T v_;

  constexpr explicit ConstExpr(T v) : v_(v) {}

  constexpr ADVENTURE_STRONG_INLINE T value_impl() const noexcept { return v_; }

  template <class Writer>
  constexpr ADVENTURE_STRONG_INLINE void derivative_impl(
      Writer &&) const noexcept {}
};

template <class Op, class L, class R, class T>
struct BinExpr : ExprBase<BinExpr<Op, L, R, T>, T> {
  using StorageType = BinExpr<Op, L, R, T>;

  typename L::StorageType lhs_;
  typename R::StorageType rhs_;
  T vLhs_;
  T vRhs_;

  template <class L_, class R_>
  BinExpr(L_ &l, R_ &r)
      : lhs_(l), rhs_(r), vLhs_(lhs_.value()), vRhs_(rhs_.value()) {}

  constexpr ADVENTURE_STRONG_INLINE T value_impl() const noexcept {
    return Op::template value<T>(vLhs_, vRhs_);
  }

  template <class Writer>
  constexpr ADVENTURE_STRONG_INLINE void derivative_impl(
      Writer &&w) const noexcept {
    struct ScaledWriter {
      Writer &w_;
      T factor_;
      constexpr ADVENTURE_STRONG_INLINE void operator()(
          index_t parent, T coeff) const noexcept {
        w_(parent, coeff * factor_);
      }
    };

    if constexpr (parent_count<L>::value != 0) {
      T left_coeff = Op::template coeff_left<T>(vLhs_, vRhs_);
      lhs_.derivative(ScaledWriter{w, left_coeff});
    }

    if constexpr (parent_count<R>::value != 0) {
      T right_coeff = Op::template coeff_right<T>(vLhs_, vRhs_);
      rhs_.derivative(ScaledWriter{w, right_coeff});
    }
  }
};

/// Lift scalars into the expression system.
template <class T>
constexpr ADVENTURE_STRONG_INLINE auto as_expr(T v) noexcept {
  return ConstExpr<T>(v);
}

template <class Op, class E, class T>
struct UnaryExpr : ExprBase<UnaryExpr<Op, E, T>, T> {
  using StorageType = UnaryExpr<Op, E, T>;

  typename E::StorageType expr_;
  T v_;

  template <class E_>
  explicit UnaryExpr(E_ &e) : expr_(e), v_(expr_.value()) {}

  constexpr ADVENTURE_STRONG_INLINE T value_impl() const noexcept {
    return Op::template value<T>(v_);
  }

  template <class Writer>
  constexpr ADVENTURE_STRONG_INLINE void derivative_impl(
      Writer &&w) const noexcept {
    if constexpr (parent_count<E>::value != 0) {
      T factor = Op::derivative(v_);
      auto scaled = [&](index_t parent, T coeff) { w(parent, coeff * factor); };
      expr_.derivative(scaled);
    }
  }
};

template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE auto operator+(const ExprBase<L, T> &l,
                                                 const ExprBase<R, T> &r) {
  return BinExpr<AddOp, L, R, T>(static_cast<const L &>(l),
                                 static_cast<const R &>(r));
}
template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE auto operator-(const ExprBase<L, T> &l,
                                                 const ExprBase<R, T> &r) {
  return BinExpr<SubOp, L, R, T>(static_cast<const L &>(l),
                                 static_cast<const R &>(r));
}
template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE auto operator*(const ExprBase<L, T> &l,
                                                 const ExprBase<R, T> &r) {
  return BinExpr<MulOp, L, R, T>(static_cast<const L &>(l),
                                 static_cast<const R &>(r));
}
template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE auto operator/(const ExprBase<L, T> &l,
                                                 const ExprBase<R, T> &r) {
  return BinExpr<DivOp, L, R, T>(static_cast<const L &>(l),
                                 static_cast<const R &>(r));
}

template <class L, class T, class U,
          std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
constexpr ADVENTURE_STRONG_INLINE auto operator/(const ExprBase<L, T> &l, U r) {
  return l / ConstExpr<T>{static_cast<T>(r)};
}
template <class R, class T, class U,
          std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
constexpr ADVENTURE_STRONG_INLINE auto operator/(U l, const ExprBase<R, T> &r) {
  return ConstExpr<T>{static_cast<T>(l)} / r;
}

template <class L, class T, class U,
          std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
constexpr ADVENTURE_STRONG_INLINE auto operator*(const ExprBase<L, T> &l, U r) {
  return l * ConstExpr<T>{static_cast<T>(r)};
}
template <class R, class T, class U,
          std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
constexpr ADVENTURE_STRONG_INLINE auto operator*(U l, const ExprBase<R, T> &r) {
  return ConstExpr<T>{static_cast<T>(l)} * r;
}

template <class L, class T, class U,
          std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
constexpr ADVENTURE_STRONG_INLINE auto operator+(const ExprBase<L, T> &l, U r) {
  return l + ConstExpr<T>{static_cast<T>(r)};
}
template <class R, class T, class U,
          std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
constexpr ADVENTURE_STRONG_INLINE auto operator+(U l, const ExprBase<R, T> &r) {
  return ConstExpr<T>{static_cast<T>(l)} + r;
}

template <class L, class T, class U,
          std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
constexpr ADVENTURE_STRONG_INLINE auto operator-(const ExprBase<L, T> &l, U r) {
  return l - ConstExpr<T>{static_cast<T>(r)};
}
template <class R, class T, class U,
          std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
constexpr ADVENTURE_STRONG_INLINE auto operator-(U l, const ExprBase<R, T> &r) {
  return ConstExpr<T>{static_cast<T>(l)} - r;
}

template <class E, class T>
constexpr ADVENTURE_STRONG_INLINE auto sin(const ExprBase<E, T> &e) {
  return UnaryExpr<SinOp, E, T>(static_cast<const E &>(e));
}

template <class E, class T>
constexpr ADVENTURE_STRONG_INLINE auto cos(const ExprBase<E, T> &e) {
  return UnaryExpr<CosOp, E, T>(static_cast<const E &>(e));
}

template <class E, class T>
constexpr ADVENTURE_STRONG_INLINE auto exp(const ExprBase<E, T> &e) {
  return UnaryExpr<ExpOp, E, T>(static_cast<const E &>(e));
}

template <class E, class T>
constexpr ADVENTURE_STRONG_INLINE auto log(const ExprBase<E, T> &e) {
  return UnaryExpr<LogOp, E, T>(static_cast<const E &>(e));
}

template <class E, class T>
constexpr ADVENTURE_STRONG_INLINE auto log10(const ExprBase<E, T> &e) {
  return UnaryExpr<Log10Op, E, T>(static_cast<const E &>(e));
}

template <class E, class T>
constexpr ADVENTURE_STRONG_INLINE auto tanh(const ExprBase<E, T> &e) {
  return UnaryExpr<TanhOp, E, T>(static_cast<const E &>(e));
}

template <class E, class T>
constexpr ADVENTURE_STRONG_INLINE auto tan(const ExprBase<E, T> &e) {
  return UnaryExpr<TanOp, E, T>(static_cast<const E &>(e));
}

template <class E, class T>
constexpr ADVENTURE_STRONG_INLINE auto abs(const ExprBase<E, T> &e) {
  return UnaryExpr<AbsOp, E, T>(static_cast<const E &>(e));
}

template <class E, class T>
constexpr ADVENTURE_STRONG_INLINE auto sqrt(const ExprBase<E, T> &e) {
  return UnaryExpr<SqrtOp, E, T>(static_cast<const E &>(e));
}

template <class E, class T>
constexpr ADVENTURE_STRONG_INLINE auto cbrt(const ExprBase<E, T> &e) {
  return UnaryExpr<CbrtOp, E, T>(static_cast<const E &>(e));
}

template <class E, class T>
constexpr ADVENTURE_STRONG_INLINE auto atan(const ExprBase<E, T> &e) {
  return UnaryExpr<AtanOp, E, T>(static_cast<const E &>(e));
}

template <class E, class T>
constexpr ADVENTURE_STRONG_INLINE auto operator-(const ExprBase<E, T> &e) {
  return UnaryExpr<NegOp, E, T>(static_cast<const E &>(e));
}

template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE auto min(const ExprBase<L, T> &l,
                                           const ExprBase<R, T> &r) {
  return BinExpr<MinOp, L, R, T>(static_cast<const L &>(l),
                                 static_cast<const R &>(r));
}
template <class L, class T>
constexpr ADVENTURE_STRONG_INLINE auto min(const ExprBase<L, T> &l, T r) {
  return min(l, as_expr(r));
}
template <class R, class T>
constexpr ADVENTURE_STRONG_INLINE auto min(T l, const ExprBase<R, T> &r) {
  return min(as_expr(l), r);
}

template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE auto max(const ExprBase<L, T> &l,
                                           const ExprBase<R, T> &r) {
  return BinExpr<MaxOp, L, R, T>(static_cast<const L &>(l),
                                 static_cast<const R &>(r));
}
template <class L, class T>
constexpr ADVENTURE_STRONG_INLINE auto max(const ExprBase<L, T> &l, T r) {
  return max(l, as_expr(r));
}
template <class R, class T>
constexpr ADVENTURE_STRONG_INLINE auto max(T l, const ExprBase<R, T> &r) {
  return max(as_expr(l), r);
}

template <class X, class L, class H, class T>
constexpr ADVENTURE_STRONG_INLINE auto clamp(const ExprBase<X, T> &x,
                                             const ExprBase<L, T> &lo,
                                             const ExprBase<H, T> &hi) {
  return max(min(x, hi), lo);
}
template <class X, class L, class T>
constexpr ADVENTURE_STRONG_INLINE auto clamp(const ExprBase<X, T> &x,
                                             const ExprBase<L, T> &lo, T hi) {
  return clamp(x, lo, as_expr(hi));
}
template <class X, class H, class T>
constexpr ADVENTURE_STRONG_INLINE auto clamp(const ExprBase<X, T> &x, T lo,
                                             const ExprBase<H, T> &hi) {
  return clamp(x, as_expr(lo), hi);
}
template <class X, class T>
constexpr ADVENTURE_STRONG_INLINE auto clamp(const ExprBase<X, T> &x, T lo,
                                             T hi) {
  return clamp(x, as_expr(lo), as_expr(hi));
}
template <class L, class H, class T>
constexpr ADVENTURE_STRONG_INLINE auto clamp(T x, const ExprBase<L, T> &lo,
                                             const ExprBase<H, T> &hi) {
  return clamp(as_expr(x), lo, hi);
}
template <class L, class T>
constexpr ADVENTURE_STRONG_INLINE auto clamp(T x, const ExprBase<L, T> &lo,
                                             T hi) {
  return clamp(as_expr(x), lo, as_expr(hi));
}
template <class H, class T>
constexpr ADVENTURE_STRONG_INLINE auto clamp(T x, T lo,
                                             const ExprBase<H, T> &hi) {
  return clamp(as_expr(x), as_expr(lo), hi);
}

template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE auto pow(const ExprBase<L, T> &l,
                                           const ExprBase<R, T> &r) {
  return BinExpr<PowOp, L, R, T>(static_cast<const L &>(l),
                                 static_cast<const R &>(r));
}
template <class L, class T, class U,
          std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
constexpr ADVENTURE_STRONG_INLINE auto pow(const ExprBase<L, T> &l, U r) {
  return pow(l, ConstExpr<T>{static_cast<T>(r)});
}
template <class R, class T, class U,
          std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
constexpr ADVENTURE_STRONG_INLINE auto pow(U l, const ExprBase<R, T> &r) {
  return pow(ConstExpr<T>{static_cast<T>(l)}, r);
}

template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator==(const ExprBase<L, T> &l,
                                                  const ExprBase<R, T> &r) {
  return l.value() == r.value();
}
template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator!=(const ExprBase<L, T> &l,
                                                  const ExprBase<R, T> &r) {
  return l.value() != r.value();
}
template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator<(const ExprBase<L, T> &l,
                                                 const ExprBase<R, T> &r) {
  return l.value() < r.value();
}
template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator<=(const ExprBase<L, T> &l,
                                                  const ExprBase<R, T> &r) {
  return l.value() <= r.value();
}
template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator>(const ExprBase<L, T> &l,
                                                 const ExprBase<R, T> &r) {
  return l.value() > r.value();
}
template <class L, class R, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator>=(const ExprBase<L, T> &l,
                                                  const ExprBase<R, T> &r) {
  return l.value() >= r.value();
}

template <class L, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator==(const ExprBase<L, T> &l,
                                                  T r) {
  return l.value() == r;
}
template <class L, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator!=(const ExprBase<L, T> &l,
                                                  T r) {
  return l.value() != r;
}
template <class L, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator<(const ExprBase<L, T> &l, T r) {
  return l.value() < r;
}
template <class L, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator<=(const ExprBase<L, T> &l,
                                                  T r) {
  return l.value() <= r;
}
template <class L, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator>(const ExprBase<L, T> &l, T r) {
  return l.value() > r;
}
template <class L, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator>=(const ExprBase<L, T> &l,
                                                  T r) {
  return l.value() >= r;
}

template <class R, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator==(T l,
                                                  const ExprBase<R, T> &r) {
  return l == r.value();
}
template <class R, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator!=(T l,
                                                  const ExprBase<R, T> &r) {
  return l != r.value();
}
template <class R, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator<(T l, const ExprBase<R, T> &r) {
  return l < r.value();
}
template <class R, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator<=(T l,
                                                  const ExprBase<R, T> &r) {
  return l <= r.value();
}
template <class R, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator>(T l, const ExprBase<R, T> &r) {
  return l > r.value();
}
template <class R, class T>
constexpr ADVENTURE_STRONG_INLINE bool operator>=(T l,
                                                  const ExprBase<R, T> &r) {
  return l >= r.value();
}

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

  auto &t = get_tape<T>();
  EdgeWriter<T> writer(t);
  e.derivative(writer);

  if (writer.used == 0) {
    return Variable<T>(primal);
  }

  // node metadata, arity = number of distinct parents
  t.arities.push_back(writer.used);
  t.adj.push_back(0);

  index_t idx = t.arities.size() - 1;
  return Variable<T>(idx, primal);
}

}  // namespace adventure
