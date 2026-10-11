# One rank exhausts the module budget while another waits in MPI. A controlled
# failure must terminate the job promptly and identify the allocation budget.
execute_process(COMMAND "${MPIEXEC}" "${NPFLAG}" 2 ${PREFLAGS} "${TEST_BINARY}" -exhaust-memory ${POSTFLAGS}
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
if("${result}" STREQUAL "0" OR "${result}" MATCHES "timeout" OR NOT "${output}${error}" MATCHES "MFTL coupling: allocation/budget failure")
  message(FATAL_ERROR "Expected a diagnosed MPI budget failure, got ${result}:\n${output}\n${error}")
endif()
