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

#include <algorithm>
#include <cmath>

namespace adventure {

struct AddOp {
    template<class T> static constexpr T coeff_left (const T&, const T&) { return T(1); }
    template<class T> static constexpr T coeff_right(const T&, const T&) { return T(1); }
    template<class T> static constexpr T value(const T& a, const T& b) { return a + b; }
};

struct SubOp {
    template<class T> static constexpr T coeff_left (const T&, const T&) { return T(1); }
    template<class T> static constexpr T coeff_right(const T&, const T&) { return T(-1); }
    template<class T> static constexpr T value(const T& a, const T& b) { return a - b; }
};

struct MulOp {
    template<class T> static constexpr T coeff_left (const T& a, const T& b) { return b; }
    template<class T> static constexpr T coeff_right(const T& a, const T& b) { return a; }
    template<class T> static constexpr T value(const T& a, const T& b) { return a * b; }
};

struct DivOp {
    template<class T> static constexpr T coeff_left (const T& a, const T& b) { return T(1) / b; }
    template<class T> static constexpr T coeff_right(const T& a, const T& b) {
        return -a / (b * b);
    }
    template<class T> static constexpr T value(const T& a, const T& b) { return a / b; }
};

struct MinOp {
    template<class T> static constexpr T value(const T& a, const T& b) { return std::min(a, b); }
    template<class T> static constexpr T coeff_left(const T& a, const T& b) { return (a < b) ? T(1) : T(0); }
    template<class T> static constexpr T coeff_right(const T& a, const T& b) { return (b < a) ? T(1) : T(0); }
};

struct MaxOp {
    template<class T> static constexpr T value(const T& a, const T& b) { return std::max(a, b); }
    template<class T> static constexpr T coeff_left(const T& a, const T& b) { return (a > b) ? T(1) : T(0); }
    template<class T> static constexpr T coeff_right(const T& a, const T& b) { return (b > a) ? T(1) : T(0); }
};

struct SinOp {
    template<class T> static constexpr T derivative(const T& a) { return std::cos(a); }
    template<class T> static constexpr T value(const T& a) { return std::sin(a); }
};

struct CosOp {
    template<class T> static constexpr T derivative(const T& a) { return -std::sin(a); }
    template<class T> static constexpr T value(const T& a) { return std::cos(a); }
};

struct ExpOp {
    template<class T> static constexpr T derivative(const T& a) { return std::exp(a); }
    template<class T> static constexpr T value(const T& a) { return std::exp(a); }
};

struct LogOp {
    template<class T> static constexpr T derivative(const T& a) { return T(1) / a; }
    template<class T> static constexpr T value(const T& a) { return std::log(a); }
};

struct Log10Op {
    // d/dx log10(x) = 1 / (x * ln(10))
    template<class T>
    static constexpr T derivative(const T& a) {
        return T(1) / (a * T(std::log(T(10))));
    }

    template<class T>
    static constexpr T value(const T& a) {
        return std::log10(a);
    }
};

struct TanhOp {
    // d/dx tanh(x) = 1 - tanh(x)^2
    template<class T>
    static constexpr T derivative(const T& a) {
        const T t = std::tanh(a);
        return T(1) - t * t;
    }

    template<class T>
    static constexpr T value(const T& a) {
        return std::tanh(a);
    }
};

struct TanOp {
    // d/dx tan(x) = 1 / cos(x)^2
    template<class T>
    static constexpr T derivative(const T& a) {
        return T(1) / (std::cos(a) * std::cos(a));
    }

    template<class T>
    static constexpr T value(const T& a) {
        return std::tan(a);
    }
};

struct AtanOp {
    // d/dx atan(x) = 1 / (1 + x^2)
    template<class T>
    static constexpr T derivative(const T& a) {
        return T(1) / (T(1) + a * a);
    }

    template<class T>
    static constexpr T value(const T& a) {
        return std::atan(a);
    }
};

struct AbsOp {
    // d/dx |a| = (a > 0) ? 1 : (a < 0) ? -1 : 0
    template<class T>
    static constexpr T derivative(const T& a) {
        return (a > T(0)) ? T(1) : ((a < T(0)) ? T(-1) : T(0));
    }

    template<class T>
    static constexpr T value(const T& a) {
        return std::abs(a);
    }
};

struct SqrtOp {
    // d/dx sqrt(x) = 0.5 / sqrt(x)
    template<class T>
    static constexpr T derivative(const T& a) {
        return T(0.5) / std::sqrt(a);
    }

    template<class T>
    static constexpr T value(const T& a) {
        return std::sqrt(a);
    }
};

struct CbrtOp {
    // d/dx cbrt(x) = 1 / (3 * cbrt(x) * cbrt(x))
    template<class T>
    static constexpr T derivative(const T& a) {
        return T(1) / (T(3) * std::cbrt(a) * std::cbrt(a));
    }

    template<class T>
    static constexpr T value(const T& a) {
        return std::cbrt(a);
    }
};

struct NegOp {
    template<class T>
    static constexpr T value(const T& a) { return -a; }

    template<class T>
    static constexpr T derivative(const T&) { return T(-1); }
};

struct PowOp {
    // d(a^b)/da = b * a^(b-1)
    template<class T>
    static constexpr T coeff_left(const T& a, const T& b) {
        return b * std::pow(a, b - T(1));
    }

    // d(a^b)/db = a^b * log(a)
    template<class T>
    static constexpr T coeff_right(const T& a, const T& b) {
        return std::pow(a, b) * std::log(a);
    }

    template<class T>
    static constexpr T value(const T& a, const T& b) { return std::pow(a, b); }
};

} // namespace adventure
