# --- MFTL unit tests -----------------------------------------------------
#
# Two flavours:
#   mftl_add_test        pure C, no MPI runtime needed
#   mftl_add_mpi_test    needs MPI_Init, launched through mpirun with N ranks
#
# Every test binary returns non-zero when any assertion fails (see test_utils.h),
# which is what CTest keys off.

function(mftl_add_test name)
  add_executable(${name} ${name}.c)
  target_link_libraries(${name} PRIVATE MFTL m)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

function(mftl_add_mpi_test name nranks)
  add_executable(${name} ${name}.c)
  target_link_libraries(${name} PRIVATE MFTL MPI::MPI_C m)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${name}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nranks}
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${name}> ${MPIEXEC_POSTFLAGS})
endfunction()

# Registers an ALREADY-DEFINED test target under a second name and rank count.
# The rank count changes both the neighbour structure and which code path is
# reachable, so the same binary at a different P is genuinely different coverage
# -- and re-using the target avoids compiling the same source twice.
function(mftl_add_mpi_test_at name target nranks)
  add_test(NAME ${name}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nranks}
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}> ${MPIEXEC_POSTFLAGS})
endfunction()

# As above, but for rank counts that exceed the cores of a development machine.
#
# A 3x3x3 grid is the smallest one with an INTERIOR rank -- 26 neighbours that
# are 26 distinct ranks -- and that property is worth having in CI even though
# 27 processes on a laptop are heavily oversubscribed and slow. Open MPI refuses
# to start more processes than slots without being told; MPICH and Intel MPI
# oversubscribe silently and reject the flag, so it is only passed when the
# launcher is recognised as Open MPI.
function(mftl_add_mpi_test_oversub name target nranks)
  set(_extra "")
  execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --version
                  OUTPUT_VARIABLE _mpi_ver ERROR_VARIABLE _mpi_ver
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
  if(_mpi_ver MATCHES "Open MPI" OR _mpi_ver MATCHES "OpenRTE")
    set(_extra "--oversubscribe")
  endif()
  add_test(NAME ${name}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nranks} ${_extra}
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}> ${MPIEXEC_POSTFLAGS})
endfunction()

# --- Layer 0: standalone utilities (no MPI) ---
mftl_add_test(test_vectorops)
mftl_add_test(test_memory)
mftl_add_test(test_arrayfunctions)

# --- Layer 1-3: everything that needs a domain, hence MPI ---
mftl_add_mpi_test(test_object 1)
mftl_add_mpi_test(test_ensemble 1)
mftl_add_mpi_test(test_geometry 1)
mftl_add_mpi_test(test_topology 1)
mftl_add_mpi_test(test_derived 1)
mftl_add_mpi_test(test_meshgen 1)
mftl_add_mpi_test(test_vertex_normals 1)

# The BVH's serial half: predicates, build, refit, traversal, and both halves of
# the periodic problem. Everything here is local by construction -- the module's
# only collective is the spatial ghost exchange, which test_bvh_parallel covers.
mftl_add_mpi_test(test_bvh 1)

# The BVH's only collective is the spatial ghost exchange, and it is registered
# at three counts because each exercises a different part of it:
#
#   2   a thin 2x1x1 grid, so the contact region lies on the single cut and both
#       bodies are split by it -- the case where every contact is cross-rank
#   4   2x2x1: the first grid with a diagonal neighbour, so a triangle can be
#       wanted by a rank that is not a face neighbour
#   8   2x2x2: every one of the 26 offsets is populated, and ranks legitimately
#       hold nothing of either body while still having to enter every collective
mftl_add_mpi_test(test_bvh_parallel 2)
mftl_add_mpi_test_at(test_bvh_parallel_np4 test_bvh_parallel 4)
mftl_add_mpi_test_at(test_bvh_parallel_np8 test_bvh_parallel 8)

# Where the LBVH meets the spatial halo. The table of placements inside is
# resolved against the ACTUAL process grid, so each rank count is a different
# geometry rather than the same test run wider:
#
#   1    no decomposition -- the reference path itself, and the guarantee that
#        every collective is still entered when there is nothing to exchange
#   2    2x1x1: every rank boundary is a FACE
#   4    2x2x1: the first grid with an EDGE, where four ranks meet on a line
#   8    2x2x2: the first grid with a CORNER, where eight meet at a point and a
#        triangle near it is wanted by seven neighbours at once
#   27   3x3x3: the first grid with an INTERIOR rank, whose 26 neighbours are 26
#        DISTINCT ranks. Below this, periodic aliasing collapses several of the
#        26 stencil offsets onto the same rank and the de-duplication is doing
#        half the work by accident.
mftl_add_mpi_test(test_bvh_ghosting 1)
mftl_add_mpi_test_at(test_bvh_ghosting_np2 test_bvh_ghosting 2)
mftl_add_mpi_test_at(test_bvh_ghosting_np4 test_bvh_ghosting 4)
mftl_add_mpi_test_at(test_bvh_ghosting_np8 test_bvh_ghosting 8)
mftl_add_mpi_test_oversub(test_bvh_ghosting_np27 test_bvh_ghosting 27)

# The remeshing OPERATORS are serial by construction -- MFTLRemeshObject asserts
# the object holds no ghosts -- so one rank is not a reduced case here, it is the
# whole domain of that file.
mftl_add_mpi_test(test_remeshing 1)

# The remeshing DRIVER is not. The shift/freeze schedule is registered at four
# counts and each covers something the others cannot:
#
#   1   the phase driver with an EMPTY frozen set. Isolates driver bugs
#       (migration, id renumbering, the closing sequence) from freeze bugs: a
#       failure here cannot be about the partition.
#   2   2x1x1. One interior plane per axis at most, and under periodicity the
#       two cut planes of the x axis are the same pair of ranks seen twice --
#       the case where a shift by h/2 maps a rank's band onto its own interior.
#   4   2x2x1: the first grid where an entity can be near the planes of TWO axes
#       at once, which is the TUBE conflict the even-parity phases exist for.
#   8   2x2x2: CORNER conflicts, i.e. entities near all three axes' planes, which
#       only the odd-parity phases resolve. Also the count at which the initial
#       scatter from rank 0 reaches non-neighbours and forces the global path.
mftl_add_mpi_test(test_remeshing_parallel 1)
mftl_add_mpi_test_at(test_remeshing_parallel_np2 test_remeshing_parallel 2)
mftl_add_mpi_test_at(test_remeshing_parallel_np4 test_remeshing_parallel 4)
mftl_add_mpi_test_at(test_remeshing_parallel_np8 test_remeshing_parallel 8)
#
#   27  3x3x3. Two things only this grid can see, and they are different
#       questions. First, at cuts[k] == 2 a periodic axis has the SAME rank on
#       both sides of a cell, so the shifted ownership rule can confuse "one cell
#       left" with "one cell right" and no test at np <= 8 notices; three is the
#       smallest count where those are different ranks. Second, h falls to 2/3 of
#       the domain, so the h > 4*delta precondition stops being comfortable and
#       the band measurement is exercised against a real margin rather than a
#       generous one.
mftl_add_mpi_test_oversub(test_remeshing_parallel_np27 test_remeshing_parallel 27)

# The OTHER distributed strategy: move the bodies instead of the planes. Same
# rank counts, and for the same reasons -- the fixture puts one droplet on every
# cut plane, so np = 2, 4 and 8 each split a different number of bodies. The
# np = 1 run is not redundant: it is where the dispersed driver's fallback must
# NOT fire, because one rank holds everything and every body is trivially local.
mftl_add_mpi_test(test_remeshing_dispersed 1)
mftl_add_mpi_test_at(test_remeshing_dispersed_np2 test_remeshing_dispersed 2)
mftl_add_mpi_test_at(test_remeshing_dispersed_np4 test_remeshing_dispersed 4)
mftl_add_mpi_test_at(test_remeshing_dispersed_np8 test_remeshing_dispersed 8)

# --- The remeshing module's MPI must stay in one file ----------------------
#
# A source-level test, because it is a source-level property and no runtime
# check can see it: every MPI call the remesher makes lives in remesh_comm.c,
# and the operators, the smoother, the compaction and both drivers contain none.
#
# That is what makes "the serial path is unaffected by the parallel work"
# checkable rather than merely intended, and it is exactly the property that was
# lost the first time round -- when the shift reached redistribution by growing
# MFTLGetOwnerProc and the id exchange by growing the per-timestep ghost sync.
# Both are cheap to reintroduce by accident and neither would fail a test.
add_test(NAME test_remeshing_mpi_is_confined
         COMMAND ${CMAKE_COMMAND}
                 -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../src/remeshing
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_mpi_confinement.cmake)

# --- Genuine multi-rank coverage ------------------------------------------
#
# At nproc == 1 the collectives are called correctly but most cross-rank
# behaviour (real neighbour discovery, periodic de-duplication on a thin
# decomposition, SUM-reducing non-zero values from more than one rank, and mesh
# generation's rank-0-only semantics) is either degenerate or entirely
# unexercised. These targets exist specifically to cover that.
mftl_add_mpi_test(test_ensemble_parallel 8)
mftl_add_mpi_test(test_meshgen_parallel 2)

# The generation contract at a third process count: a thin 4x1x1 decomposition
# is where periodic wrap-around maps several of the 26 neighbour offsets onto the
# same rank, which is the case the de-duplication in domain_init_comm exists for.
mftl_add_mpi_test_at(test_meshgen_parallel_np4 test_meshgen_parallel 4)

# The per-body reduction: two routes to one answer, plus a third written in the
# test to check both against. The rank counts are chosen by what each REACHES,
# and the reasons are in the file banner -- 2 and 4 are the grids whose
# comm_graph carries a self-loop, 8 is the first with a corner, and 27 is the
# first with an interior rank whose 26 neighbours are 26 distinct ranks.
mftl_add_mpi_test(test_body_reduce 1)
mftl_add_mpi_test_at(test_body_reduce_np2 test_body_reduce 2)
mftl_add_mpi_test_at(test_body_reduce_np4 test_body_reduce 4)
mftl_add_mpi_test_at(test_body_reduce_np8 test_body_reduce 8)
mftl_add_mpi_test_oversub(test_body_reduce_np27 test_body_reduce 27)

# Redistribution is the one module whose behaviour is a FUNCTION of the process
# grid, so it is registered at four counts and each is genuinely different:
#
#   1   the no-op path (and the guarantee that it really is a no-op)
#   2   a thin 2x1x1 grid: periodic wrap maps several of the 26 neighbour offsets
#       onto the same rank, so the neighbour list is shorter than the stencil
#   4   2x2x1: the first grid with a real diagonal neighbour
#   8   2x2x2: every one of the 26 offsets is populated, and the initial scatter
#       from rank 0 reaches ranks that are not its neighbours -- which is what
#       forces the global communication path
mftl_add_mpi_test(test_redistribution 1)
mftl_add_mpi_test_at(test_redistribution_np2 test_redistribution 2)
mftl_add_mpi_test_at(test_redistribution_np4 test_redistribution 4)
mftl_add_mpi_test_at(test_redistribution_np8 test_redistribution 8)

# I/O is registered at three counts, and each covers something different:
#
#   1   the piece IS the whole mesh, so the file totals must match the mesh
#       exactly -- there is no seam duplication to explain a discrepancy away
#   2   two pieces to concatenate, and a shared file whose exclusive scan is
#       non-trivial on exactly one rank
#   4   2x2x1: bodies split across diagonal neighbours, and ranks that
#       legitimately hold nothing and must still enter every collective write
mftl_add_mpi_test(test_io 1)
mftl_add_mpi_test_at(test_io_np2 test_io 2)
mftl_add_mpi_test_at(test_io_np4 test_io 4)

# MFTLComputeVertexNormals is a LOCAL operation, so a second rank count is not
# about communication -- it is about the empty-object path. MFTLGenerateSphere
# gives rank 0 the whole mesh, so at nproc == 3 the other two ranks run the pass
# on a pool with no allocation at all, and any collective accidentally introduced
# into it would deadlock here rather than in production.
mftl_add_mpi_test_at(test_vertex_normals_np3 test_vertex_normals 3)

# The scaling benchmark is an EXAMPLE, and no scaling sweep belongs in this
# suite -- the build guide (MFTL_Documentation/pages/building.md) states the
# rule: "The rank count is not a scaling knob in this suite; it changes which
# code path is reachable."
#
# What is registered is a CORRECTNESS check of that example, because a program
# whose whole purpose is to report timings has to be right before its numbers
# mean anything. Rigid per-body translation conserves surface area exactly and
# redistribution conserves entities exactly, so --verify needs no tolerance
# beyond round-off, and the example exits non-zero when it drifts. np = 4 is the
# smallest grid with a real diagonal neighbour.
if(BUILD_EXAMPLES)
  add_test(NAME test_advection_scaling_conserves
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:parallel_advection_scaling>
                   --spheres 8 --refine 1 --steps 3 --warmup 1 --verify ${MPIEXEC_POSTFLAGS})

  # The SOLVER-SIDE half of the same rule, and it is the half no library test can
  # reach: a caller that holds an MFTL_Object* across a redistribution advects a
  # pointer the redistribution invalidated. Every structural invariant survives
  # it -- the mesh stays a valid closed sphere with good triangles -- and the
  # enclosed volume is the only thing that moves, so nothing short of a physical
  # measurement over several timesteps sees it.
  #
  # np = 2 because it is invisible at np = 1: nothing is ever destroyed there,
  # and the same code is correct. Twelve steps is enough for the failure to be a
  # factor of thirteen rather than a tolerance question.
  add_test(NAME test_leveque_parallel_conserves
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:leveque_remeshing_parallel>
                   -steps 12 -budget 4 -quiet -verify ${MPIEXEC_POSTFLAGS})
endif()
