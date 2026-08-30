# adventure - A Minimal Reverse-Mode Automatic Differentiation Library for C++

![CI](https://github.com/buech/adventure/actions/workflows/ci-linux.yml/badge.svg)

adventure is a fast and lightweight C++20 library providing reverse-mode
automatic differentiation using Jacobi tapes and expression templates.

## Features
- first-order reverse-mode derivatives using Jacobi tapes
- expression templates
- thread-local tapes
- header-only
- C++20

## Quick Start
- Include `adventure/variable.hpp` and use the `adventure::Variable` API:

```cpp
#include <iostream>

#include "adventure/variable.hpp"

namespace ad = adventure;

int main() {
  auto &tape = ad::get_tape<double>();
  ad::Variable<double> x(2.0);

  tape.register_input(x);
  ad::Variable<double> y = x * x;
  tape.register_output(y);

  y.grad() = 1.0;          // seed output
  tape.backward();         // compute adjoints
  double dx = x.grad();    // gradient dy/dx

  std::cout << "x.grad() = " << dx << std::endl;
}
```
## Building and Running Tests and Benchmarks
### Build
Configure and build in a separate directory:

```bash
cmake -S . -B build
cmake --build build
```

### Unit Tests
After building, run the tests with `ctest`:

```bash
ctest --test-dir build --output-on-failure
```

### Benchmarks
Benchmarks are optional but enabled by default. After building, run the
benchmark executables in `build/benchmarks/` (e.g. `benchmark_ad`).

## License
adventure is licensed under the Apache License, Version 2.0 (see
[LICENSE-APACHE](LICENSE-APACHE)), or, at your option, the MIT License (see
[LICENSE-MIT](LICENSE-MIT)).
