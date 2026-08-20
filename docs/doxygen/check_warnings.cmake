# Fails if Doxygen's warning log is non-empty.
#
# Doxygen exits 0 even when it emits warnings, so "warnings are the acceptance
# criterion" cannot be enforced on the exit code. This script is the POST_BUILD
# step of the `docs` target when MFTL_DOCS_WARN_AS_ERROR is ON, and it is the
# CMake counterpart of `python3 build_docs.py --strict`.
#
# Invoked as:  cmake -DLOG=<path> -P check_warnings.cmake

if(NOT DEFINED LOG)
    message(FATAL_ERROR "check_warnings.cmake: -DLOG=<path> is required")
endif()

if(NOT EXISTS "${LOG}")
    # No log at all means Doxygen never wrote one, which is the clean case.
    return()
endif()

file(READ "${LOG}" _contents)
string(STRIP "${_contents}" _stripped)

if(_stripped STREQUAL "")
    return()
endif()

string(REPLACE "\n" ";" _lines "${_stripped}")
list(LENGTH _lines _count)

message(STATUS "----------------------------------------------------------")
message(STATUS "Doxygen emitted ${_count} warning(s):")
message(STATUS "----------------------------------------------------------")

set(_shown 0)
foreach(_line IN LISTS _lines)
    if(_shown LESS 25)
        message(STATUS "  ${_line}")
        math(EXPR _shown "${_shown} + 1")
    endif()
endforeach()

if(_count GREATER 25)
    math(EXPR _rest "${_count} - 25")
    message(STATUS "  ... and ${_rest} more")
endif()

message(FATAL_ERROR
    "MFTL_DOCS_WARN_AS_ERROR is ON and the documentation is not clean. "
    "Full log: ${LOG}")
