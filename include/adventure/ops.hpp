/*
 * Copyright (c) 2025 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#pragma once

#include <algorithm>
#include <cmath>

namespace adventure {

enum class OpCode : std::uint8_t {
  LEAF,  // Input / Independent variable
  CONSTANT,
  COPY,  // y = x
  ADD,   // y = a + b
  SUB,   // y = a - b
  MUL,   // y = a * b
  DIV,   // y = a / b
  MIN,
  MAX,
  SIN,  // y = sin(a)
  COS,  // y = cos(a)
  EXP,  // y = exp(a)
  LOG,  // y = log(a)
  LOG10,
  TANH,
  TAN,
  ATAN,
  ABS,
  SQRT,
  QBRT,
  NEG,
  POW,
  // ... other unary and binary ops
};

struct AddOp {
  static constexpr OpCode opcode = OpCode::ADD;
  template <class T>
  static constexpr T coeff_left(const T &, const T &) {
    return T(1);
  }
  template <class T>
  static constexpr T coeff_right(const T &, const T &) {
    return T(1);
  }
  template <class T>
  static constexpr T value(const T &a, const T &b) {
    return a + b;
  }
};

struct SubOp {
  static constexpr OpCode opcode = OpCode::SUB;
  template <class T>
  static constexpr T coeff_left(const T &, const T &) {
    return T(1);
  }
  template <class T>
  static constexpr T coeff_right(const T &, const T &) {
    return T(-1);
  }
  template <class T>
  static constexpr T value(const T &a, const T &b) {
    return a - b;
  }
};

struct MulOp {
  static constexpr OpCode opcode = OpCode::MUL;

  template <class T>
  static constexpr T coeff_left(const T &a, const T &b) {
    return b;
  }
  template <class T>
  static constexpr T coeff_right(const T &a, const T &b) {
    return a;
  }
  template <class T>
  static constexpr T value(const T &a, const T &b) {
    return a * b;
  }
};

struct DivOp {
  static constexpr OpCode opcode = OpCode::DIV;

  template <class T>
  static constexpr T coeff_left(const T &a, const T &b) {
    return T(1) / b;
  }
  template <class T>
  static constexpr T coeff_right(const T &a, const T &b) {
    return -a / (b * b);
  }
  template <class T>
  static constexpr T value(const T &a, const T &b) {
    return a / b;
  }
};

struct MinOp {
  static constexpr OpCode opcode = OpCode::MIN;

  template <class T>
  static constexpr T value(const T &a, const T &b) {
    return std::min(a, b);
  }
  template <class T>
  static constexpr T coeff_left(const T &a, const T &b) {
    return (a < b) ? T(1) : T(0);
  }
  template <class T>
  static constexpr T coeff_right(const T &a, const T &b) {
    return (b < a) ? T(1) : T(0);
  }
};

struct MaxOp {
  static constexpr OpCode opcode = OpCode::MAX;

  template <class T>
  static constexpr T value(const T &a, const T &b) {
    return std::max(a, b);
  }
  template <class T>
  static constexpr T coeff_left(const T &a, const T &b) {
    return (a > b) ? T(1) : T(0);
  }
  template <class T>
  static constexpr T coeff_right(const T &a, const T &b) {
    return (b > a) ? T(1) : T(0);
  }
};

struct SinOp {
  static constexpr OpCode opcode = OpCode::SIN;

  template <class T>
  static constexpr T derivative(const T &a) {
    return std::cos(a);
  }
  template <class T>
  static constexpr T value(const T &a) {
    return std::sin(a);
  }
};

struct CosOp {
  static constexpr OpCode opcode = OpCode::COS;

  template <class T>
  static constexpr T derivative(const T &a) {
    return -std::sin(a);
  }
  template <class T>
  static constexpr T value(const T &a) {
    return std::cos(a);
  }
};

struct ExpOp {
  static constexpr OpCode opcode = OpCode::EXP;

  template <class T>
  static constexpr T derivative(const T &a) {
    return std::exp(a);
  }
  template <class T>
  static constexpr T value(const T &a) {
    return std::exp(a);
  }
};

struct LogOp {
  static constexpr OpCode opcode = OpCode::LOG;

  template <class T>
  static constexpr T derivative(const T &a) {
    return T(1) / a;
  }
  template <class T>
  static constexpr T value(const T &a) {
    return std::log(a);
  }
};

struct Log10Op {
  static constexpr OpCode opcode = OpCode::LOG10;

  // d/dx log10(x) = 1 / (x * ln(10))
  template <class T>
  static constexpr T derivative(const T &a) {
    return T(1) / (a * T(std::log(T(10))));
  }

  template <class T>
  static constexpr T value(const T &a) {
    return std::log10(a);
  }
};

struct TanhOp {
  static constexpr OpCode opcode = OpCode::TANH;

  // d/dx tanh(x) = 1 - tanh(x)^2
  template <class T>
  static constexpr T derivative(const T &a) {
    const T t = std::tanh(a);
    return T(1) - t * t;
  }

  template <class T>
  static constexpr T value(const T &a) {
    return std::tanh(a);
  }
};

struct TanOp {
  static constexpr OpCode opcode = OpCode::TAN;

  // d/dx tan(x) = 1 / cos(x)^2
  template <class T>
  static constexpr T derivative(const T &a) {
    return T(1) / (std::cos(a) * std::cos(a));
  }

  template <class T>
  static constexpr T value(const T &a) {
    return std::tan(a);
  }
};

struct AtanOp {
  static constexpr OpCode opcode = OpCode::ATAN;

  // d/dx atan(x) = 1 / (1 + x^2)
  template <class T>
  static constexpr T derivative(const T &a) {
    return T(1) / (T(1) + a * a);
  }

  template <class T>
  static constexpr T value(const T &a) {
    return std::atan(a);
  }
};

struct AbsOp {
  static constexpr OpCode opcode = OpCode::ABS;

  // d/dx |a| = (a > 0) ? 1 : (a < 0) ? -1 : 0
  template <class T>
  static constexpr T derivative(const T &a) {
    return (a > T(0)) ? T(1) : ((a < T(0)) ? T(-1) : T(0));
  }

  template <class T>
  static constexpr T value(const T &a) {
    return std::abs(a);
  }
};

struct SqrtOp {
  static constexpr OpCode opcode = OpCode::SQRT;

  // d/dx sqrt(x) = 0.5 / sqrt(x)
  template <class T>
  static constexpr T derivative(const T &a) {
    return T(0.5) / std::sqrt(a);
  }

  template <class T>
  static constexpr T value(const T &a) {
    return std::sqrt(a);
  }
};

struct CbrtOp {
  static constexpr OpCode opcode = OpCode::QBRT;

  // d/dx cbrt(x) = 1 / (3 * cbrt(x) * cbrt(x))
  template <class T>
  static constexpr T derivative(const T &a) {
    return T(1) / (T(3) * std::cbrt(a) * std::cbrt(a));
  }

  template <class T>
  static constexpr T value(const T &a) {
    return std::cbrt(a);
  }
};

struct NegOp {
  static constexpr OpCode opcode = OpCode::NEG;

  template <class T>
  static constexpr T value(const T &a) {
    return -a;
  }

  template <class T>
  static constexpr T derivative(const T &) {
    return T(-1);
  }
};

struct PowOp {
  static constexpr OpCode opcode = OpCode::POW;

  // d(a^b)/da = b * a^(b-1)
  template <class T>
  static constexpr T coeff_left(const T &a, const T &b) {
    return b * std::pow(a, b - T(1));
  }

  // d(a^b)/db = a^b * log(a)
  template <class T>
  static constexpr T coeff_right(const T &a, const T &b) {
    return std::pow(a, b) * std::log(a);
  }

  template <class T>
  static constexpr T value(const T &a, const T &b) {
    return std::pow(a, b);
  }
};

}  // namespace adventure
