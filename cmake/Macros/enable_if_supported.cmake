# Macro : enable_if_supported
# Provides macro ENABLE_IF_SUPPORTED, which checks whether the CXX compiler
# understands a given compiler flag. If so, it is added to the given variable.
#

MACRO(ENABLE_IF_SUPPORTED var flag)
    STRING(REGEX REPLACE "^[-//]" "" fName "${flag}")
    check_cxx_compiler_flag("${flag}" AI_CONTAINS_FLAG_${fName})
    
    IF(AI_CONTAINS_FLAG_${fName})
        SET(${var} "${${var}} ${flag}")
    ENDIF()
ENDMACRO()
