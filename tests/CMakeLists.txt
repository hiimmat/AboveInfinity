cmake_minimum_required(VERSION 3.15)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.2
)

FetchContent_MakeAvailable(Catch2)

add_executable(ntensor_tests)

target_sources(ntensor_tests
    PRIVATE
    src/main_tests.cpp
    src/test_aligned_allocator.cpp
    src/test_bounds.cpp
    src/test_dense_buffer.cpp
    src/test_dimensions.cpp
    src/test_execute.cpp
    src/test_plane.cpp
    src/test_planes.cpp
    src/test_range.cpp
    src/test_reshape.cpp
    src/test_shape_transmutation.cpp
    src/test_sparse_buffer.cpp
    src/test_stream_io.cpp
    src/test_strides.cpp
    src/test_tensor.cpp
   src/test_utilities.cpp
)

target_include_directories(ntensor_tests
    PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_link_libraries(ntensor_tests PRIVATE Catch2::Catch2WithMain)

target_compile_features(ntensor_tests PRIVATE cxx_std_20)

add_compile_definitions(NT_ALIGNMENT=128)

target_compile_options(
  ntensor_tests
  PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/std:c++20>
          $<$<CXX_COMPILER_ID:MSVC>:/W4>
          $<$<CXX_COMPILER_ID:MSVC>:/WX>
          $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
          $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-pedantic>)

add_test(NAME ntensor_tests COMMAND ntensor_tests)
