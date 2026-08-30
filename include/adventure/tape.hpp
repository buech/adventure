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
#include <limits>

#include "adventure/config.hpp"
#include "adventure/dynamic_array.hpp"

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
 private:
  Tape() = default;

 public:
  /// Type used for tape indices.
  using index_type = index_t;

  /// Arities (number of unique expression arguments) per node.
  internal::DynamicArray<std::uint8_t> arities;
  /// Edges per node.
  internal::DynamicArray<Edge<T>> edges;
  /// Adjoint values per node.
  internal::DynamicArray<T> adj;

  static Tape &get_tape() {
    static ADVENTURE_THREAD_LOCAL Tape tape;
    return tape;
  }

  Tape(const Tape &) = delete;
  void operator=(const Tape &) = delete;

  /// Get the current tape size (number of nodes).
  std::size_t size() const noexcept { return arities.size(); }

  /// Maximum representable index value.
  static constexpr index_type max_index =
      std::numeric_limits<index_type>::max();

  ADVENTURE_STRONG_INLINE index_type add_leaf() noexcept {
    // The index that this node will receive.
    std::size_t next = arities.size();
    assert(next <= static_cast<std::size_t>(max_index) &&
           "adventure::Tape overflow - index exceeds the maximum value of "
           "ADVENTURE_INDEX_TYPE");

    // Record node metadata.
    arities.push_back(static_cast<std::uint8_t>(0));
    adj.push_back(0);

    return static_cast<index_type>(next);
  }

  ADVENTURE_STRONG_INLINE index_type add_unary(index_type p, T c) noexcept {
    // The index that this node will receive.
    std::size_t next = arities.size();
    assert(next <= static_cast<std::size_t>(max_index) &&
           "adventure::Tape overflow - index exceeds the maximum value of "
           "ADVENTURE_INDEX_TYPE");

    // Store edges.
    edges.emplace_back(p, c);

    // Record node metadata.
    arities.push_back(static_cast<std::uint8_t>(1));
    adj.push_back(0);

    return static_cast<index_type>(next);
  }

  template <typename U>
  void register_input(U &var) {
    var.idx = add_leaf();
  }

  template <typename U>
  void register_output(U &var) {
    if (!var.is_active())
      var.idx = add_leaf();
    else
      var.idx = add_unary(var.idx, T(1));
  }

  /// Clear all recorded operations.
  void clear() noexcept {
    arities.clear();
    edges.clear();
    adj.clear();
  }

  void reserve(std::size_t nNodes, std::size_t nEdges) noexcept {
    arities.reserve(nNodes);
    edges.reserve(nEdges);
    adj.reserve(nNodes);
  }

  void backward() noexcept {
    const std::size_t node_cnt = arities.size();
    std::size_t edge = edges.size();  // one past last edge

    for (std::size_t i = node_cnt; i-- > 0;) {
      const auto ar = arities[i];
      edge -= ar;  // now points to the first edge of node i

      const T seed = adj[i];
      if (seed == T(0)) [[unlikely]]
        continue;
      // walk the edge block backwards
      for (auto k = ar; k-- > 0;) {
        const auto &e = edges[edge + k];
        adj[e.parent] += e.coeff * seed;
      }
    }
  }

  T &adj_at(index_type idx) noexcept { return adj[idx]; }
};

template <typename T>
Tape<T> &get_tape() {
  return Tape<T>::get_tape();
}

}  // namespace adventure
