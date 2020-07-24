# AboveInfinity

AboveInfinity is a templated C++17 header-only library used for tensor manipulation. The current version of the library is 0.2. It can be built with or without CMake. If it's built with CMake, the minimum required version of CMake is 3.12.

This library provides a number of data structures and data descriptors used to easier represent multidimensional data (Tensors and TensorViews), as well as several functionalities that allow manipulating and/or iterating over the same.

## Features

This library contains the implementation of Tensor as its main data structure and TensorView as the main data descriptor. All the features provided by this library were designed to work with Tensors and/or TensorViews independently of their rank (with rank 1 Tensors/TensorViews being the exception when it comes to functions such as `slice` and `slab`, since there is no support for rank 0 Tensors/TensorViews). This means that this library doesn't enforce a limit on the Tensor's rank to be functional.

The following is a brief selection of the most important features includes:
- fastPermute (permutes the dimensions of a TensorView)
- undoPermutation (undoes the permutation done through fastPermute to a TensorView)
- slice (performs a hyperplane over a given dimension of a TensorView)
- slab (performs a hyperslab over a given dimension of a TensorView)
- slicingPointer (retrieves the pointer to the data of a TensorView that one would get after performing several slice operations)
- subspace (a combination of slice and slab inspired by NumPy's indexing)
- newAxis (increases the dimension of a TensorView by one)
- squeeze (removes all single dimensions from a TensorView)
- reshape (changes the shape of a TensorView without affecting the stored data)
- flatten (copies the data a TensorView points to into a rank 1 Tensor)
- ravel (attempts to convert a TensorView into a rank 1 view. If unable, it copies the data into a rank 1 Tensor)
- saveAsText (saves a TensorView into a text file - used in combination with loadFromText)
- loadFromText (loads the data from a text file into a TensorView - used in combination with saveAsText)
- execute (executes a function over one or more TensorViews)
- copy (copies the data from one TensorView into another TensorView)

## Building & Compatibility

The whole code can be found in the `include` directory, and that's the only thing one needs to use it. Optionally, this library can be included through CMake, and there's an example on how to do it in the "examples" directory. The difference between using CMake and not using CMake is that the CMake code generates one more file named `SysInfo.h`. Also, if CMake is used, a few other options can be defined through it. Although, the code already has default values for each one of them. Moreover, note that the current code doesn't make use of the variables available through `SysInfo.h` since those were meant for a future version.

The latest version of the code is tested exclusively on Windows 10 using GCC 7.3.0. Previous versions were tested both on Windows 10 and Ubuntu 18.04., in both cases using GCC (version 7.3.0. on Windows, and 9.3. on Ubuntu).

An much earlier version supported msvc, however, there were several issues that lead me to drop the support for it. I dropped the support for msvc after getting error `C1035` when trying to compile the function `subspace` that can be found in `include/TensorView.h`.
Hopefully, a future version will add support for msvc again, as well as Clang.

## Further changes

As already mentioned, there are plans for future versions that would include further functionality additions and improvements, testing refactors, documentation, more examples, and hopefully wider compiler and architecture support. However, those largely depend on my spare time that can be used for this project. As such, the possible release dates remain unknown.

Functionalities that are planned for the next release include:
- Matrix multiplication
- Einsum with optimized contractions through a combinatorial optimization algorithm and a brute-force algorithm 
<br />
Comments and bug reports are welcome. 
