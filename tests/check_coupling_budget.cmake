# One rank exhausts the module budget while another waits in MPI. A controlled
# failure must terminate the job promptly and identify the allocation budget.
# The failing rank writes its diagnostic to LOG_FILE: the launcher's stderr
# forwarding can lose it when the job is torn down.
cmake_policy(VERSION 3.13)  # a -P script starts with every policy unset
file(REMOVE "${LOG_FILE}")
execute_process(COMMAND "${MPIEXEC}" "${NPFLAG}" 2 ${PREFLAGS} "${TEST_BINARY}" -exhaust-memory "${LOG_FILE}" ${POSTFLAGS}
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
set(diagnostic "")
if(EXISTS "${LOG_FILE}")
  file(READ "${LOG_FILE}" diagnostic)
endif()
if("${result}" STREQUAL "0" OR "${result}" MATCHES "timeout" OR NOT "${diagnostic}" MATCHES "MFTL coupling: allocation/budget failure")
  message(FATAL_ERROR "Expected a diagnosed MPI budget failure, got ${result}:\n${diagnostic}\n${output}\n${error}")
endif()
