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

#include <cstddef>
#include <memory>
#include <type_traits>
#include <cstring> // for std::memcpy
#include <utility>
#include <cassert>
#include <cstdlib>

#include "adventure/config.hpp"

namespace adventure::internal {

struct ExponentialGrowth { };
struct NoGrowth { };

template<class T, class GrowthStrategy = ExponentialGrowth>
class DynamicArray {
    static_assert(std::is_trivially_destructible_v<T>,
                  "DynamicArray only supports trivially-destructible types");

public:
    using value_type      = T;
    using size_type       = std::size_t;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;

    constexpr DynamicArray() noexcept = default;

    explicit DynamicArray(std::size_t capacity) { reserve(capacity); }

    DynamicArray(const DynamicArray&)            = delete;
    DynamicArray& operator=(const DynamicArray&) = delete;

    DynamicArray(DynamicArray&& other) noexcept
        : raw_buf_(std::move(other.raw_buf_)),
          buf_(other.buf_),
          sz_(other.sz_),
          cap_(other.cap_)
    {
        other.buf_ = nullptr;
        other.sz_  = 0;
        other.cap_ = 0;
    }

    DynamicArray& operator=(DynamicArray&& other) noexcept {
        if (this != &other) {
            raw_buf_ = std::move(other.raw_buf_);
            buf_     = other.buf_;
            sz_      = other.sz_;
            cap_     = other.cap_;

            other.buf_ = nullptr;
            other.sz_  = 0;
            other.cap_ = 0;
        }
        return *this;
    }

    ~DynamicArray() noexcept = default; // unique_ptr frees the memory

    constexpr size_type size() const noexcept { return sz_; }
    constexpr size_type capacity() const noexcept { return cap_; }
    constexpr bool empty() const noexcept { return sz_ == 0; }

    /// Ensure that at least `n` elements can be stored.
    /**
     *  If `n` > current capacity a new buffer is allocated,
     *  the old data is copied with a single `memcpy`,
     *  and the old buffer is released.
     */
    void reserve(size_type n) {
        if (n <= cap_) return;

        // allocate a new block
        std::size_t bytes = n * sizeof(T);
        void* raw = std::malloc(bytes);
        if (!raw) throw std::bad_alloc();

        T* new_buf = reinterpret_cast<T*>(raw);

        // copy existing data (trivially copyable -> memcpy)
        if (sz_ > 0) {
            std::memcpy(new_buf, buf_, sz_ * sizeof(T));
        }

        // release the old buffer (unique_ptr takes care of it)
        raw_buf_.reset(reinterpret_cast<unsigned char*>(raw));
        buf_   = new_buf;
        cap_   = n;
    }

    void ensure_capacity() {
        if constexpr (!std::is_same_v<GrowthStrategy, NoGrowth>) {
            if (sz_ == cap_) [[unlikely]] {
                size_type new_cap = cap_ ? cap_ * 2 : 1;
                reserve(new_cap);
            }
        }
    }

    ADVENTURE_STRONG_INLINE void push_back(const T& v) noexcept {
        ensure_capacity();
        new (&buf_[sz_]) T(v);
        ++sz_;
    }

    ADVENTURE_STRONG_INLINE void push_back(T&& v) noexcept {
        ensure_capacity();
        new (&buf_[sz_]) T(std::move(v));
        ++sz_;
    }

    template<class... Args>
    ADVENTURE_STRONG_INLINE T& emplace_back(Args&&... args) noexcept {
        ensure_capacity();
        new (&buf_[sz_]) T(std::forward<Args>(args)...);
        ++sz_;
        return buf_[sz_ - 1];
    }

    void clear() noexcept { sz_ = 0; }

    void resize(size_type n) noexcept {
        if (n > cap_) reserve(n);
        if (n > sz_) {
            for (size_type i = sz_; i < n; ++i) {
                new (&buf_[i]) T();
            }
        }
        // For trivially-destructible types we can just shrink the size.
        sz_ = n;
    }

    ADVENTURE_STRONG_INLINE reference operator[](size_type i) noexcept {
        assert(i < sz_);
        return buf_[i];
    }
    ADVENTURE_STRONG_INLINE const_reference operator[](size_type i) const noexcept {
        assert(i < sz_);
        return buf_[i];
    }

    pointer data() noexcept { return buf_; }
    const_pointer data() const noexcept { return buf_; }

private:
    std::unique_ptr<unsigned char, decltype(&std::free)> raw_buf_{nullptr, &std::free};
    T*        buf_   = nullptr;
    size_type sz_    = 0;
    size_type cap_   = 0;
};

} // namespace adventure::internal
