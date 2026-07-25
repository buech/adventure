/*
 * Copyright (c) 2025 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#pragma once

#include <cstdint>
#include <type_traits>

#ifndef ADVENTURE_INDEX_TYPE
#define ADVENTURE_INDEX_TYPE std::uint32_t
#endif

#ifndef ADVENTURE_STRONG_INLINE
#define ADVENTURE_STRONG_INLINE inline __attribute__((always_inline))
#endif

#ifndef ADVENTURE_THREAD_LOCAL
#define ADVENTURE_THREAD_LOCAL \
  thread_local __attribute__((tls_model("initial-exec")))
#endif

#ifndef ADVENTURE_SHALLOW_COPY
#define ADVENTURE_SHALLOW_COPY
#endif

namespace adventure {

/// The concrete type that the library will use for tape indices.
using index_t = ADVENTURE_INDEX_TYPE;

// Compile-time sanity check. The index must be an unsigned integral type.
static_assert(std::is_integral_v<index_t>,
              "ADVENTURE_INDEX_TYPE must be an integral type");
static_assert(!std::is_signed_v<index_t>,
              "ADVENTURE_INDEX_TYPE must be unsigned (signed indices make "
              "overflow detection ambiguous)");

}  // namespace adventure
