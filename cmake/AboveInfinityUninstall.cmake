################ CMake Uninstall Template #######################
# CMake Template file for uninstallation of files
# mentioned in 'install_manifest.txt'
#
# Used by uninstall target
#################################################################

set(MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt")

if(EXISTS ${MANIFEST})
  message(STATUS "============== Uninstalling AboveInfinity  ===================")

  file(STRINGS ${MANIFEST} files)
  foreach(file ${files})
    if(EXISTS ${file})
      message(STATUS "Removing file: '${file}'")

      execute_process(
        COMMAND ${CMAKE_COMMAND} -E remove ${file}
        OUTPUT_VARIABLE rm_out
        RESULT_VARIABLE rm_retval
        )

      if(NOT "${rm_retval}" STREQUAL 0)
        message(FATAL_ERROR "Failed to remove file: '${file}'.")
      endif()
    else()
      message(STATUS "File '${file}' does not exist.")
    endif()
  endforeach()

  message(STATUS "========== Finished Uninstalling AboveInfinity  ==============")
else()
  message(STATUS "Cannot find install manifest: '${MANIFEST}'")
  message(STATUS "make install has probably not been performed")
  message(STATUS "   or install_manifest.txt has been deleted.")
endif()
