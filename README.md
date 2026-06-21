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
  ad::Variable<double> x(2.0);

  ad::register_input(x);
  ad::Variable<double> y = x * x;
  ad::register_output(y);

  y.grad() = 1.0;          // seed output
  ad::backward<double>();  // compute adjoints
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
adventure is licensed under the Apache License, Version
2.0 (see [LICENSE](LICENSE)).

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
