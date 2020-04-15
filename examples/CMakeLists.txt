cmake_minimum_required(VERSION 3.12)

project("AboveInfinity-Example")

find_package(AboveInfinity CONFIG REQUIRED)

add_executable(${PROJECT_NAME} src/source.cc)
target_link_libraries(${PROJECT_NAME} AboveInfinity::AboveInfinity)