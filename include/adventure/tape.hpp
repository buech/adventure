/*
 * Copyright (c) 2025 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

#include "adventure/config.hpp"
#include "adventure/dynamic_array.hpp"
#include "adventure/ops.hpp"

namespace adventure {

template <typename T>
struct Edge {
  index_t parent;
  T coeff;
};

/// The tape that records all operations for a thread.
/**
 * @tparam T Scalar type.
 */
template <typename T>
class Tape {
 public:
  Tape() = default;

  /// Type used for tape indices.
  using index_type = index_t;

  internal::DynamicArray<T> primals;
  internal::DynamicArray<T> adj;
  internal::DynamicArray<OpCode> ops;
  internal::DynamicArray<index_type> operands;

  static Tape &get_tape() {
    static ADVENTURE_THREAD_LOCAL Tape tape;
    return tape;
  }

  Tape(const Tape &) = delete;
  void operator=(const Tape &) = delete;

  /// Get the current tape size (number of nodes).
  std::size_t size() const noexcept { return primals.size(); }

  /// Maximum representable index value.
  static constexpr index_type max_index =
      std::numeric_limits<index_type>::max();

  index_type add_constant(T val) noexcept {
    index_type idx = static_cast<index_type>(primals.size());
    assert(idx <= static_cast<std::size_t>(max_index) &&
           "adventure::Tape overflow - index exceeds the maximum value of "
           "ADVENTURE_INDEX_TYPE");
    primals.push_back(val);
    adj.push_back(T(0));
    ops.push_back(OpCode::CONSTANT);
    return idx;
  }

  index_type add_leaf(T primal) noexcept {
    index_type idx = static_cast<index_type>(primals.size());
    assert(idx <= static_cast<std::size_t>(max_index) &&
           "adventure::Tape overflow - index exceeds the maximum value of "
           "ADVENTURE_INDEX_TYPE");
    primals.push_back(primal);
    adj.push_back(T(0));
    ops.push_back(OpCode::LEAF);
    // operands.push_back(idx);
    return idx;
  }

  index_type add_unary(OpCode op, index_type parent, T primal) noexcept {
    index_type idx = static_cast<index_type>(primals.size());
    assert(idx <= static_cast<std::size_t>(max_index) &&
           "adventure::Tape overflow - index exceeds the maximum value of "
           "ADVENTURE_INDEX_TYPE");
    primals.push_back(primal);
    adj.push_back(T(0));
    ops.push_back(op);
    operands.push_back(parent);
    return idx;
  }

  index_type add_binary(OpCode op, index_type lhs, index_type rhs,
                        T primal) noexcept {
    index_type idx = static_cast<index_type>(primals.size());
    assert(idx <= static_cast<std::size_t>(max_index) &&
           "adventure::Tape overflow - index exceeds the maximum value of "
           "ADVENTURE_INDEX_TYPE");
    primals.push_back(primal);
    adj.push_back(T(0));
    ops.push_back(op);
    operands.push_back(lhs);
    operands.push_back(rhs);
    return idx;
  }

  template <typename U>
  index_t register_input(U &var) {
    var.idx = add_leaf(var.value());
    var.tape_ptr = this;
    return var.idx;
  }

  template <typename U>
  index_t register_output(U &var) {
    if (!var.is_active())
      var.idx = add_leaf(var.value());
    else
      var.idx = add_unary(OpCode::COPY, var.idx, var.value());
    var.tape_ptr = this;
    return var.idx;
  }

  void reset_adj() noexcept {
    for (std::size_t i = 0; i < adj.size(); ++i) adj[i] = T(0);
  }

  /// Clear all recorded operations.
  void clear() noexcept {
    primals.clear();
    adj.clear();
    ops.clear();
    operands.clear();
  }

  void reserve(std::size_t nNodes, std::size_t nEdges) noexcept {
    primals.reserve();
    adj.reserve();
    ops.reserve();
    operands.reserve();
  }

  void dump() noexcept {
    for (std::size_t i = 0; i < primals.size(); ++i) {
      std::cout << "prim[" << i << "] = " << primals[i] << '\n';
      std::cout << " adj[" << i << "] = " << adj[i] << '\n';
      std::cout << "  op[" << i << "] = " << int(ops[i]) << '\n';
    }
    std::cout << "------------------" << '\n';
    for (std::size_t i = 0; i < operands.size(); ++i) {
      std::cout << "operands[" << i << "] = " << operands[i] << '\n';
    }
    std::cout << "==================" << '\n';
  }

  static constexpr std::uint8_t op_arity(OpCode op) noexcept {
    switch (op) {
      case OpCode::CONSTANT:
      case OpCode::LEAF:
        return 0;
      case OpCode::COPY:
      case OpCode::NEG:
      case OpCode::SIN:
      case OpCode::COS:
      case OpCode::EXP:
      case OpCode::LOG:
      case OpCode::LOG10:
      case OpCode::TANH:
      case OpCode::TAN:
      case OpCode::ATAN:
      case OpCode::ABS:
      case OpCode::SQRT:
      case OpCode::QBRT:
        return 1;
      case OpCode::ADD:
      case OpCode::SUB:
      case OpCode::MUL:
      case OpCode::DIV:
      case OpCode::MIN:
      case OpCode::MAX:
      case OpCode::POW:
        return 2;
    }
    return 0;
  }

  void primal() noexcept {
    if (primals.empty()) return;

    std::size_t arg_idx = 0;

    for (std::size_t i = 0; i < ops.size(); ++i) {
      switch (ops[i]) {
        case OpCode::CONSTANT:
        case OpCode::LEAF:
          break;

        case OpCode::COPY: {
          index_t p = operands[arg_idx++];
          primals[i] = primals[p];
          break;
        }
        case OpCode::ADD: {
          index_t lhs = operands[arg_idx++];
          index_t rhs = operands[arg_idx++];
          primals[i] = primals[lhs] + primals[rhs];
          break;
        }
        case OpCode::MUL: {
          index_t lhs = operands[arg_idx++];
          index_t rhs = operands[arg_idx++];
          primals[i] = primals[lhs] * primals[rhs];
          break;
        }
        case OpCode::SIN: {
          index_t p = operands[arg_idx++];
          primals[i] = std::sin(primals[p]);
          break;
        }
        case OpCode::COS: {
          index_t p = operands[arg_idx++];
          primals[i] = std::cos(primals[p]);
          break;
        }
        case OpCode::EXP: {
          index_t p = operands[arg_idx++];
          primals[i] = std::exp(primals[p]);
          break;
        }
        case OpCode::LOG: {
          index_t p = operands[arg_idx++];
          primals[i] = std::log(primals[p]);
          break;
        }
        case OpCode::LOG10: {
          index_t p = operands[arg_idx++];
          primals[i] = std::log10(primals[p]);
          break;
        }
        case OpCode::TANH: {
          index_t p = operands[arg_idx++];
          primals[i] = std::tanh(primals[p]);
          break;
        }
        case OpCode::TAN: {
          index_t p = operands[arg_idx++];
          primals[i] = std::tan(primals[p]);
          break;
        }
        case OpCode::ATAN: {
          index_t p = operands[arg_idx++];
          primals[i] = std::atan(primals[p]);
          break;
        }
        case OpCode::ABS: {
          index_t p = operands[arg_idx++];
          primals[i] = std::abs(primals[p]);
          break;
        }
        case OpCode::SQRT: {
          index_t p = operands[arg_idx++];
          primals[i] = std::sqrt(primals[p]);
          break;
        }
        case OpCode::QBRT: {
          index_t p = operands[arg_idx++];
          primals[i] = std::cbrt(primals[p]);
          break;
        }
        case OpCode::NEG: {
          index_t p = operands[arg_idx++];
          primals[i] = -primals[p];
          break;
        }
        case OpCode::SUB: {
          index_t lhs = operands[arg_idx++];
          index_t rhs = operands[arg_idx++];
          primals[i] = primals[lhs] - primals[rhs];
          break;
        }
        case OpCode::DIV: {
          index_t lhs = operands[arg_idx++];
          index_t rhs = operands[arg_idx++];
          primals[i] = primals[lhs] / primals[rhs];
          break;
        }
        case OpCode::MIN: {
          index_t lhs = operands[arg_idx++];
          index_t rhs = operands[arg_idx++];
          primals[i] = std::min(primals[lhs], primals[rhs]);
          break;
        }
        case OpCode::MAX: {
          index_t lhs = operands[arg_idx++];
          index_t rhs = operands[arg_idx++];
          primals[i] = std::max(primals[lhs], primals[rhs]);
          break;
        }
        case OpCode::POW: {
          index_t lhs = operands[arg_idx++];
          index_t rhs = operands[arg_idx++];
          primals[i] = std::pow(primals[lhs], primals[rhs]);
          break;
        }
      }
    }

    assert(arg_idx == operands.size() &&
           "Operands buffer was not consumed completely!");
  }

  void backward() noexcept {
    if (primals.empty()) return;

    std::size_t arg_idx = operands.size();  // points one past the end

    for (std::size_t i = primals.size(); i-- > 0;) {
      const T seed = adj[i];
      const OpCode op = ops[i];

      if (seed == T(0)) {
        // Even if seed == 0, remember to step the cursor past this node's
        // operands!
        arg_idx -= op_arity(op);
        continue;
      }

      switch (op) {
        case OpCode::CONSTANT:
        case OpCode::LEAF:
          break;

        case OpCode::COPY: {
          const index_type p = operands[--arg_idx];
          adj[p] += seed;
          break;
        }

        case OpCode::SIN: {
          const index_type p = operands[--arg_idx];
          adj[p] += std::cos(primals[p]) * seed;
          break;
        }

        case OpCode::COS: {
          const index_type p = operands[--arg_idx];
          adj[p] -= std::sin(primals[p]) * seed;
          break;
        }

        case OpCode::EXP: {
          const index_type p = operands[--arg_idx];
          adj[p] += std::exp(primals[p]) * seed;
          break;
        }

        case OpCode::LOG: {
          const index_type p = operands[--arg_idx];
          adj[p] += seed / primals[p];
          break;
        }

        case OpCode::LOG10: {
          const index_type p = operands[--arg_idx];
          adj[p] += seed / (primals[p] * std::log(T(10)));
          break;
        }

        case OpCode::TANH: {
          const index_type p = operands[--arg_idx];
          const T t = std::tanh(primals[p]);
          adj[p] += (T(1) - t * t) * seed;
          break;
        }

        case OpCode::TAN: {
          const index_type p = operands[--arg_idx];
          const T c = std::cos(primals[p]);
          adj[p] += seed / (c * c);
          break;
        }

        case OpCode::ATAN: {
          const index_type p = operands[--arg_idx];
          adj[p] += seed / (T(1) + primals[p] * primals[p]);
          break;
        }

        case OpCode::ABS: {
          const index_type p = operands[--arg_idx];
          const T a = primals[p];
          adj[p] += (a > T(0) ? T(1) : (a < T(0) ? T(-1) : T(0))) * seed;
          break;
        }

        case OpCode::SQRT: {
          const index_type p = operands[--arg_idx];
          adj[p] += (T(0.5) / std::sqrt(primals[p])) * seed;
          break;
        }

        case OpCode::QBRT: {
          const index_type p = operands[--arg_idx];
          const T c = std::cbrt(primals[p]);
          adj[p] += seed / (T(3) * c * c);
          break;
        }

        case OpCode::NEG: {
          const index_type p = operands[--arg_idx];
          adj[p] -= seed;
          break;
        }

        case OpCode::ADD: {
          // Pushed in order (lhs, rhs), so stepping backwards:
          const index_type rhs = operands[--arg_idx];
          const index_type lhs = operands[--arg_idx];
          adj[lhs] += seed;
          adj[rhs] += seed;
          break;
        }

        case OpCode::SUB: {
          const index_type rhs = operands[--arg_idx];
          const index_type lhs = operands[--arg_idx];
          adj[lhs] += seed;
          adj[rhs] -= seed;
          break;
        }

        case OpCode::MUL: {
          const index_type rhs = operands[--arg_idx];
          const index_type lhs = operands[--arg_idx];
          adj[lhs] += primals[rhs] * seed;
          adj[rhs] += primals[lhs] * seed;
          break;
        }

        case OpCode::DIV: {
          const index_type rhs = operands[--arg_idx];
          const index_type lhs = operands[--arg_idx];
          const T inv_rhs = T(1) / primals[rhs];
          adj[lhs] += inv_rhs * seed;
          adj[rhs] -= (primals[i] * inv_rhs) * seed;
          break;
        }

        case OpCode::MIN: {
          const index_type rhs = operands[--arg_idx];
          const index_type lhs = operands[--arg_idx];
          if (primals[lhs] < primals[rhs])
            adj[lhs] += seed;
          else if (primals[rhs] < primals[lhs])
            adj[rhs] += seed;
          break;
        }

        case OpCode::MAX: {
          const index_type rhs = operands[--arg_idx];
          const index_type lhs = operands[--arg_idx];
          if (primals[lhs] > primals[rhs])
            adj[lhs] += seed;
          else if (primals[rhs] > primals[lhs])
            adj[rhs] += seed;
          break;
        }

        case OpCode::POW: {
          const index_type rhs = operands[--arg_idx];
          const index_type lhs = operands[--arg_idx];
          const T a = primals[lhs];
          const T b = primals[rhs];
          adj[lhs] += (b * std::pow(a, b - T(1))) * seed;
          adj[rhs] += (primals[i] * std::log(a)) * seed;
          break;
        }
      }
    }

    assert(arg_idx == 0 && "Operands buffer was not consumed completely!");
  }

  const T &adj_at(index_type idx) const noexcept { return adj[idx]; }
  T &adj_at(index_type idx) noexcept { return adj[idx]; }
};

template <typename T>
Tape<T> &get_tape() {
  return Tape<T>::get_tape();
}

}  // namespace adventure
