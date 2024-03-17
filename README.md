# Introduction

ntensor is a C++20 template header only library for multi-dimensional array manipulation. The library is inspired by [NumPy](https://numpy.org/), a Python library for scientific computing.

The main container in this library is a Tensor, which consists of a variadic number of planes, each representing a multidimensional array. Each plane can have one or more channels, allowing the Tensor to be regular, interlaved, semi-interleaved, or planar. Furthermore, each plane can have a distinct shape and a distinct buffer used for memory allocation.

Additionally, Tensors adopt a compile-time variadic version of the policy design pattern, which allows for easier addition/removal of features from the container itself.

# Usage

### Create a tensor with a single 1D plane, and reshape the plane into a 3D plane

```
#include <dense_buffer.hpp>
#include <plane.hpp>
#include <reshape.hpp>
#include <shape_transmutation.hpp>
#include <tensor.hpp>

namespace nt = ntensor;

int main() {
  static constexpr nt::Dimensions<24u> dimensions;
  auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
  auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);

  static constexpr nt::Dimensions<4u, 2u, 3u> reshaped_dimensions;
  auto reshaped_tensor = nt::reshape<reshaped_dimensions>(tensor);

  return 0;
}
```

### Create a tensor with three planes, where the second and third planes have half the size of the first plane

```
#include <dense_buffer.hpp>
#include <plane.hpp>
#include <shape_transmutation.hpp>
#include <tensor.hpp>

namespace nt = ntensor;

int main() {
  static constexpr nt::Dimensions<4u, 3u> dimensions1;
  static constexpr nt::Dimensions<2u, 3u> dimensions2;

  auto plane1 = nt::create_plane<nt::DenseBuffer<int>, dimensions1>();
  auto plane2 = nt::create_plane<nt::DenseBuffer<int>, dimensions2>();
  auto plane3 = nt::create_plane<nt::DenseBuffer<int>, dimensions2>();

  auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane1, plane2, plane3);

  return 0;
}
```

### Initialize the elements of a tensor with random values in a range [0, 100]
```
#include <dense_buffer.hpp>
#include <execute.hpp>
#include <plane.hpp>
#include <random>
#include <shape_transmutation.hpp>
#include <tensor.hpp>

namespace nt = ntensor;

int main() {
  static constexpr nt::Dimensions<4u, 2u, 3u> dimensions;
  auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
  auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  auto generate_number = [&distribution, &generator](int& e) { e = distribution(generator); };

  nt::execute(generate_number, tensor);

  return 0;
}
```

### Find the element with the largest value inside the tensor

```
#include <dense_buffer.hpp>
#include <execute.hpp>
#include <plane.hpp>
#include <random>
#include <shape_transmutation.hpp>
#include <tensor.hpp>

namespace nt = ntensor;

int main() {
  static constexpr nt::Dimensions<4u, 2u, 3u> dimensions;
  auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
  auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  auto generate_number = [&distribution, &generator](int& e) { e = distribution(generator); };

  nt::execute(generate_number, tensor);

  int largest_value = 0;

  auto find_largest_value = [&largest_value](int& e) { e > largest_value ? largest_value = e : 0; };

  nt::execute(find_largest_value, tensor);

  std::cout << largest_value << std::endl;

  return 0;
}
```

### Index access

```
#include <dense_buffer.hpp>
#include <plane.hpp>
#include <shape_transmutation.hpp>
#include <tensor.hpp>

namespace nt = ntensor;

int main() {
  static constexpr nt::Dimensions<4u, 2u, 3u> dimensions;
  auto plane = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
  auto tensor = nt::create_tensor<nt::ShapeTransmutation>(plane);

  // Access a specific element inside the Tensor.
  // For example access the element at index [3][1][2]
  // The first value passed to the function is the channel
  auto& value = tensor.slicing_value(0u, 3u, 1u, 2u);

  // Create a slice of the Tensor.
  // For example, turn [4][2][3] into [4][3]
  auto slice = tensor.slice<1u>(1u);

  // Create a slab of the Tensor
  // For example, turn [4][2][3] into [2]][2][3]
  auto slab = tensor.slab<0u, 1u, 3u>();

  // Combination of using slab and slice
  // For example, turn [4][2][3] into [2][3]
  // The first value passed to the function is the plane index
  auto subspace = tensor.template subspace<0u, nt::range<0u, 2u>{}, nt::range<2u, 2u>{}, nt::range<0u, 3u>{}>();

  return 0;
}
```

### Write the tensor to a sink/load the tensor from a source

```
#include <dense_buffer.hpp>
#include <execute.hpp>
#include <plane.hpp>
#include <random>
#include <shape_transmutation.hpp>
#include <sstream>
#include <stream_io.hpp>
#include <tensor.hpp>

namespace nt = ntensor;

int main() {
  static constexpr nt::Dimensions<4u, 2u, 3u> dimensions;
  auto plane1 = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
  auto tensor1 = nt::create_tensor<nt::ShapeTransmutation>(plane1);

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  auto generate_number = [&distribution, &generator](int& e) { e = distribution(generator); };

  nt::execute(generate_number, tensor1);

  std::stringstream ss;

  nt::write_to_sink(tensor1, ss);

  auto plane2 = nt::create_plane<nt::DenseBuffer<int>, dimensions>();
  auto tensor2 = nt::create_tensor<nt::ShapeTransmutation>(plane2);

  nt::load_from_source(tensor2, ss.view());

  return 0;
}
```


# Building and Installation
To build the unit tests, the [Catch2](https://github.com/catchorg/Catch2) unit testing framework is required. If it's not already available on the system, CMake will attempt to download it.

The library has no other dependencies. So in case if the tests aren't needed, the files can be included as a standalone project.

Another option is to install the library or add it as a sub-directory through CMake with tests enabled/disabled.

## cmake installation

```
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=<installation_path>
cmake --build . --target install
```

## cmake sub-directory

ntensor can be included using `add_subdirectory` if it cannot be installed on a system.

If ntensor is included and the `EXCLUDE_FROM_ALL` flag is set, ntensor tests can be disabled by setting the `NTENSOR_BUILD_TESTS` option to OFF.

```
 set(NTENSOR_BUILD_TESTS OFF)
  add_subdirectory(${ntensor_SOURCE_DIR} ${ntensor_BINARY_DIR}
                   EXCLUDE_FROM_ALL)
```
