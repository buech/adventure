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
