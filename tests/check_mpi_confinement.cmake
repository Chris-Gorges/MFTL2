# Fails if any source in SRC_DIR outside the ALLOWED files calls MPI.
#
# Used for the modules with networking of their own -- src/remeshing/ (ALLOWED
# defaults to remesh_comm.{c,h}), src/topology_change/ (topo_comm.c),
# src/coupling/ (coupling_comm.c) -- and for modules that have none:
# src/correction/ and the computational modules, with ALLOWED=NONE. SRC_DIR may
# be a comma-separated list. What follows was written for the remesher and holds
# for the others for the same reason.
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
#
# The rule once bred wrappers: four modules each kept a private in-place
# Allreduce and a private MPI_Wtime, only to keep them out of their drivers. The
# library has one of each (MFTLAllreduceDoubles/Ints/LongLongs and MFTLWallTime,
# src/communication/collectives.c), so those two spellings are refused in EVERY
# scanned file, the allowed ones included.

# A `cmake -P` script starts with every policy unset, so under CMake 3.x the
# IN_LIST operator below is not an operator at all and the script errors out;
# CMake 4 dropped the old behaviour, which is why this only failed off the
# development machine. Pin the policies to the version the project requires.
cmake_policy(VERSION 3.13)

if(NOT DEFINED SRC_DIR)
  message(FATAL_ERROR "check_mpi_confinement: SRC_DIR not set")
endif()

string(REPLACE "," ";" _dirs "${SRC_DIR}")
set(_sources "")
foreach(_d IN LISTS _dirs)
  if(NOT IS_DIRECTORY "${_d}")
    message(FATAL_ERROR "check_mpi_confinement: ${_d} is not a directory")
  endif()
  file(GLOB _in_dir "${_d}/*.c" "${_d}/*.h")
  if(NOT _in_dir)
    message(FATAL_ERROR "check_mpi_confinement: no sources in ${_d}, so the check would pass vacuously")
  endif()
  list(APPEND _sources ${_in_dir})
endforeach()

if(DEFINED ALLOWED)
  string(REPLACE "," ";" _allowed "${ALLOWED}")
else()
  set(_allowed "remesh_comm.c" "remesh_comm.h")
endif()
if(_allowed STREQUAL "NONE")
  set(_allowed "")
endif()

set(_offenders "")
set(_shared "")

foreach(_f IN LISTS _sources)
  get_filename_component(_name "${_f}" NAME)

  # The library's shared primitives, spelled privately: refused everywhere.
  file(STRINGS "${_f}" _private REGEX "MPI_Wtime[ \t]*\\(|MPI_Allreduce[ \t]*\\([ \t]*MPI_IN_PLACE")
  foreach(_line IN LISTS _private)
    string(REGEX MATCH "^[ \t]*(\\*|//|/\\*)" _is_comment "${_line}")
    if(_is_comment STREQUAL "")
      list(APPEND _shared "${_name}: ${_line}")
    endif()
  endforeach()

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

if(_shared)
  message("")
  foreach(_o IN LISTS _shared)
    message("  ${_o}")
  endforeach()
  message(FATAL_ERROR
          "A private wall clock or in-place Allreduce in ${SRC_DIR}. Use the library's "
          "MFTLWallTime() and MFTLAllreduceDoubles/Ints/LongLongs() (mftl_communication.h).")
endif()

if(_offenders)
  message("")
  foreach(_o IN LISTS _offenders)
    message("  ${_o}")
  endforeach()
  if(_allowed)
    list(GET _allowed 0 _home)
    message(FATAL_ERROR
            "MPI calls found outside ${_home} in ${SRC_DIR}. Every collective and every "
            "point-to-point the module makes belongs in that file; see its header for why.")
  else()
    message(FATAL_ERROR
            "MPI calls found in ${SRC_DIR}, which makes none of its own. Communicate "
            "through the library's primitives (mftl_communication.h, mftl_ensemble.h).")
  endif()
endif()

list(LENGTH _sources _n_sources)
message(STATUS "MPI confinement of ${SRC_DIR}: OK (${_n_sources} sources)")
