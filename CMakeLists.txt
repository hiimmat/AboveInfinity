# MSVC's Cmake server mode had a bug reading the following format:
# https://github.com/fmtlib/fmt/issues/809
cmake_minimum_required(VERSION 3.12...3.17)

# else needed because of the previous mentioned issue in "cmake_minimum_required"
if(${CMAKE_VERSION} VERSION_LESS 3.17)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.17)
endif()

project(
  "AboveInfinity"
  VERSION 0.1
  DESCRIPTION "A header only C++ library that provides a set of templated data structures with a number of features and standalone functions for easier multidimensional data manipulation."
  HOMEPAGE_URL "https://github.com/hiimmat/AboveInfinity"
  LANGUAGES CXX)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory and run CMake from there. You may need to remove the CMakeCache.txt file. ")
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

include(GNUInstallDirs)

set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
  CACHE PATH "The directory relative to CMAKE_PREFIX_PATH where this projects header files are installed")
set(CMAKEPACKAGE_INSTALL_DIR
  "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake"
  CACHE PATH "The directory relative to CMAKE_PREFIX_PATH where this projects config.cmake file is installed")
set(ABOVEINFINITY_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${INCLUDE_INSTALL_DIR}")
set(ABOVEINFINITY_ROOT_DIR ${CMAKE_INSTALL_PREFIX})

option(ENABLE_REQUIREMENTS "Enable static asserts" ON)
set(MAX_STACK_ALLOC_SIZE "1024" CACHE STRING "Maximum size in bytes of a Tensor that can be allocated on the stack, default is 1024")

foreach(key
  IN ITEMS
    NUMBER_OF_LOGICAL_CORES
    NUMBER_OF_PHYSICAL_CORES
    HOSTNAME
    FQDN
    TOTAL_VIRTUAL_MEMORY
    AVAILABLE_VIRTUAL_MEMORY
    TOTAL_PHYSICAL_MEMORY
    AVAILABLE_PHYSICAL_MEMORY
    IS_64BIT
    HAS_FPU
    HAS_MMX
    HAS_MMX_PLUS
    HAS_SSE
    HAS_SSE2
    HAS_SSE_FP
    HAS_SSE_MMX
    HAS_AMD_3DNOW
    HAS_AMD_3DNOW_PLUS
    HAS_IA64
    HAS_SERIAL_NUMBER
    PROCESSOR_SERIAL_NUMBER
    PROCESSOR_NAME
    PROCESSOR_DESCRIPTION
    OS_NAME
    OS_RELEASE
    OS_VERSION
    OS_PLATFORM
  )
  cmake_host_system_information(RESULT _${key} QUERY ${key})
endforeach()

configure_file(cmake/sys_info.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/SysInfo.h @ONLY)

include (TestBigEndian)
TEST_BIG_ENDIAN(IS_BIG_ENDIAN)

if(ENABLE_REQUIREMENTS)
  add_compile_definitions(ENABLE_REQUIREMENTS)
endif()


string(TOLOWER "${CMAKE_GENERATOR}" cmake_generator_tolower)
if(cmake_generator_tolower MATCHES "makefile")
  message(STATUS "Some things you can do now:")
  message(STATUS "Command       |   Description")
  message(STATUS "make install  | Install AboveInfinity. Headers will be installed to:")
  message(STATUS "              |     <CMAKE_INSTALL_PREFIX>/<INCLUDE_INSTALL_DIR>")
  message(STATUS "              |   Using the following values:")
  message(STATUS "              |     CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "              |     INCLUDE_INSTALL_DIR:  ${INCLUDE_INSTALL_DIR}")
  message(STATUS "              |   Change the install location of the AboveInfinity headers using:")
  message(STATUS "              |     cmake . -DCMAKE_INSTALL_PREFIX=prefix")
  message(STATUS "              |   Or:")
  message(STATUS "              |     cmake . -DINCLUDE_INSTALL_DIR=dir")
  message(STATUS "make uninstall| Removes files installed by make install")
endif()

message(STATUS "")

target_compile_definitions(${PROJECT_NAME} INTERFACE _MAXSTACKALLOCSIZE = MAX_STACK_ALLOC_SIZE _IS_BIG_ENDIAN=IS_BIG_ENDIAN)

target_include_directories(
  ${PROJECT_NAME}
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME ${PROJECT_NAME})
install (TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets)
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  PATH_VARS ABOVEINFINITY_INCLUDE_DIR ABOVEINFINITY_ROOT_DIR
  INSTALL_DESTINATION ${CMAKEPACKAGE_INSTALL_DIR})

write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY ExactVersion
  ARCH_INDEPENDENT)

export(TARGETS ${PROJECT_NAME}
       NAMESPACE ${PROJECT_NAME}::
       FILE ${PROJECT_NAME}Targets.cmake)

export (PACKAGE ${PROJECT_NAME})

install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKEPACKAGE_INSTALL_DIR})

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKEPACKAGE_INSTALL_DIR})

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
        DESTINATION ${CMAKEPACKAGE_INSTALL_DIR})

# Add uninstall target
add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Uninstall.cmake)
