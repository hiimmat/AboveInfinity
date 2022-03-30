# AboveInfinity

AboveInfinity is a templated C++17 header-only library used for tensor manipulation. The current version of the library is 0.2. It can be built with or without CMake. If it's built with CMake, the minimum required version of CMake is 3.12.

This library provides data structures and data descriptors used to easier represent multidimensional data (Tensors and TensorViews), as well as functionalities that allow manipulating and/or iterating over the same.

## Features

This library contains the implementation of Tensor as its main data structure and TensorView as the main data descriptor. All the features provided by this library were designed to work with Tensors and/or TensorViews independently of their rank.

There's also an implementation of a newer version of the Tensor class that can work both as Tensors and TensorViews. It can be found in the experimental directory since there's still no full support for it (it's tested, but there are no written test cases for it and the TensorOps methods need to be rewritten to support it). However, some advantages of that class are that it now supports different memory layouts, such as real, interleaved, semi-interleaved, and planar layouts. It can work with sparse or dense buffers, and it's easily adjustable to work with purely compile-time and/or runtime methods. Also, adding new functionalities is quite straightforward since the class uses variadic policy-based design.

The following is a brief selection of the most important features included in this project:
- fastPermute (permutes the dimensions of a TensorView)
- undoPermutation (undoes the permutation done through fastPermute)
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

The main part of the code resides in the `include` directory, and it can be included as a standalone project. Optionally, this library can be included through CMake, and there's an example of how to do it in the `examples` directory. The difference between using CMake and not using CMake is that CMake allows us to overwrite the defaults of a few features that are used in the code.

The latest version of the code was tested on Windows 10 using GCC 7.3.0., and Ubuntu 18.04. using GCC 11.2.

An earlier version supported msvc. However, there were several reasons that lead me to drop the support for it. Although, the main reason was due to getting the error `C1035` when trying to compile the function `subspace`.

## Further changes

There were plans for newer versions which would include rewriting the existing data structures and implementing additional functionalities such as einsum, lazy evaluation and broadcasting. However, the cost-to-benefit ratio doesn't justify the continuation of this project, so for now, I'll be shifting my focus towards other projects. And maybe I'll revisit this one once C++20 becomes more stable.

<<<<<<< Updated upstream
Functionalities that are planned for the next release include:
- Einsum with optimized contractions through a combinatorial optimization algorithm and a brute-force algorithm
- Lazy evaluation
- Refactoring Tensor and TensorView into a single class with a policy-based design for memory allocations and a compile-time visitor
- Classes TensorPool and PoolTensor
=======
>>>>>>> Stashed changes
<br />
Comments and bug reports are welcome. 
