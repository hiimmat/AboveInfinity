# AboveInfinity

AboveInfinity is an header-only library, based on C++17. If used with cmake, the minimum CMake version required is 3.12.
This library provides a number of data structures used to easier represent multidimensional data (using Tensors and TensorViews), and several functionalities that allow manipulating and/or iterating over the same.
The latest version of the code is tested exclusively on Windows using GCC 7.3.0. Previous versions were tested both on Windows and on Ubuntu 18.04., both using GCC (version 7.3.0. on Windows, and 9.3. on Ubuntu).
An much earlier version supported msvc, however, there were several issues that lead me to drop the support for it. I dropped the support for msvc after getting an error "C1035" when trying to compile the function `subspace` that can be found in "include/TensorView.h".
Hopefully, a future version will add support for msvc again, as well as Clang.

As already mentioned, there are plans for future versions that would include further functionality changes, add proper testing and documentation, more examples, and hopefully wider compiler and architecture support. However, those depend largely on my spare time that can be used for this project. As such, the possible release dates remain unknown.

Comments and bug reports are welcome. 
