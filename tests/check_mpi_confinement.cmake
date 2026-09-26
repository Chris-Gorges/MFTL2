# Fails if any source in SRC_DIR outside the ALLOWED files calls MPI.
#
# Used for two modules: src/remeshing/ (ALLOWED defaults to remesh_comm.{c,h})
# and src/topology_change/ (ALLOWED=topo_comm.c). What follows was written for
# the first and holds for the second for the same reason.
#
# The remeshing module reaches the network through exactly one file. Everything
# else -- the operators, the smoother, the compaction, and both distributed
# drivers -- is local computation over a partition somebody else arranged, and
# keeping it that way is what makes the serial path provably unaffected by the
# parallel work.
#
# It is a source check because it is a source property: a driver that called
# MPI_Allreduce directly would still pass every runtime test in the suite, and
# the two intrusions this rule was written after (a cut-plane offset inside
# MFTLGetOwnerProc, a third payload inside the per-timestep ghost sync) were
# both of exactly that shape -- correct, tested, and in the wrong file.
#
# Matches `MPI_` as a whole word so that prose mentioning MPI in a comment is
# fine but a call is not. The allowed file is named rather than pattern-matched,
# so adding a second networking file is a deliberate edit here.

if(NOT DEFINED SRC_DIR)
  message(FATAL_ERROR "check_mpi_confinement: SRC_DIR not set")
endif()

file(GLOB _sources "${SRC_DIR}/*.c" "${SRC_DIR}/*.h")

if(DEFINED ALLOWED)
  string(REPLACE "," ";" _allowed "${ALLOWED}")
else()
  set(_allowed "remesh_comm.c" "remesh_comm.h")
endif()
list(GET _allowed 0 _home)
set(_offenders "")

foreach(_f IN LISTS _sources)
  get_filename_component(_name "${_f}" NAME)
  if(_name IN_LIST _allowed)
    continue()
  endif()

  file(STRINGS "${_f}" _hits REGEX "MPI_[A-Za-z_]+[ \t]*\\(")
  foreach(_line IN LISTS _hits)
    # A line that is entirely a comment is prose about MPI, not a call.
    string(REGEX MATCH "^[ \t]*(\\*|//|/\\*)" _is_comment "${_line}")
    if(_is_comment STREQUAL "")
      list(APPEND _offenders "${_name}: ${_line}")
    endif()
  endforeach()
endforeach()

if(_offenders)
  message("")
  foreach(_o IN LISTS _offenders)
    message("  ${_o}")
  endforeach()
  message(FATAL_ERROR
          "MPI calls found outside ${SRC_DIR}/${_home}. Every collective and every "
          "point-to-point the module makes belongs in that file; see its header for why.")
endif()

message(STATUS "MPI confinement of ${SRC_DIR}: OK")
