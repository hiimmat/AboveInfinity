cmake_minimum_required(VERSION 3.15)

project("ntensor"
        VERSION 0.3.0
        DESCRIPTION "A header-only C++ library for tensor manipulation"
        LANGUAGES CXX
)

if(CMAKE_PROJECT_NAME STREQUAL ${PROJECT_NAME})
  set(NTENSOR_IS_ROOT_PROJECT TRUE)
else()
  set(NTENSOR_IS_ROOT_PROJECT FALSE)
endif()

option(ENABLE_NT_EXPECTS "Enable Expects" TRUE)
option(ENABLE_NT_ENSURES "Enable Ensures" TRUE)
option(NTENSOR_BUILD_TESTS "Build ntensor tests" TRUE)
option(NTENSOR_INSTALL_LIBRARY "Install tensor library; default is disabled if ntensor is included using add_subdirectory" ${NTENSOR_IS_ROOT_PROJECT})

set(NT_ALIGNMENT 128 CACHE STRING "Memory alignment used for Tensor allocation, default is 128")

if(NT_ALIGNMENT LESS 64 OR NT_ALIGNMENT GREATER 1024)
    message(FATAL_ERROR "Invalid memory alignment specified. The allowed range is [64, 1024]")
endif()

add_library(${PROJECT_NAME} INTERFACE)

# add alias so the project can be uses with add_subdirectory
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(
    ${PROJECT_NAME}
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

target_compile_definitions(${PROJECT_NAME} INTERFACE "$<$<BOOL:${ENABLE_NT_EXPECTS}>:ENABLE_NT_EXPECTS>")
target_compile_definitions(${PROJECT_NAME} INTERFACE "$<$<BOOL:${ENABLE_NT_ENSURES}>:ENABLE_NT_ENSURES>")
target_compile_definitions(${PROJECT_NAME} INTERFACE NT_ALIGNMENT=${NT_ALIGNMENT})

if(NTENSOR_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(NTENSOR_INSTALL_LIBRARY)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    write_basic_package_version_file(
        "${PROJECT_NAME}ConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMinorVersion
    )

    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )

    install(EXPORT ${PROJECT_NAME}_Targets
            FILE ${PROJECT_NAME}Targets.cmake
            NAMESPACE ${PROJECT_NAME}::
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )

    install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )

    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/license.txt")

    include(CPack)
endif()
