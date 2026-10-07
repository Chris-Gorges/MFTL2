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

# Marching cubes: the generated 256-case table, exhaustively (crack freedom over
# every pair of cells that can share a face), and the extraction against
# analytic shapes and adversarial random / checkerboard fields. Serial by
# construction -- the lattice identity it tests is what the distributed
# topology change relies on, and that is tested there.
mftl_add_mpi_test(test_marching_cubes 1)

# The signed distance rasteriser, whose SIGN is checked against the generalised
# winding number -- a reference that shares nothing with the closest-point and
# pseudo-normal machinery -- on shapes built to break it.
mftl_add_mpi_test(test_signed_distance 1)

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

# SELF-proximity (params.self_pairs), the query a topology change is detected
# by, tested as a multiset against an exhaustive sweep taken before
# partitioning. Same rank counts and the same reasons as test_bvh_ghosting; the
# fixtures sit where the cut planes meet and the cross-rank pairs are counted.
mftl_add_mpi_test(test_bvh_self 1)
mftl_add_mpi_test_at(test_bvh_self_np2 test_bvh_self 2)
mftl_add_mpi_test_at(test_bvh_self_np4 test_bvh_self 4)
mftl_add_mpi_test_at(test_bvh_self_np8 test_bvh_self 8)
mftl_add_mpi_test_oversub(test_bvh_self_np27 test_bvh_self 27)

# Topology changes. Every scenario is checked for what the result must BE and,
# at np > 1, against a genuine np = 1 run of the same scenario on MPI_COMM_SELF
# inside the same test: same bodies, same ids, same vertex positions. The rank
# counts are the usual ones -- a face (2), an edge (4), a corner (8), an
# interior rank (27) -- and every fixture sits where the cut planes meet.
mftl_add_mpi_test(test_topology_change 1)
mftl_add_mpi_test_at(test_topology_change_np2 test_topology_change 2)
mftl_add_mpi_test_at(test_topology_change_np4 test_topology_change 4)
mftl_add_mpi_test_at(test_topology_change_np8 test_topology_change 8)
mftl_add_mpi_test_oversub(test_topology_change_np27 test_topology_change 27)

# The remeshing OPERATORS are serial by construction -- MFTLRemeshObject asserts
# the object holds no ghosts -- so one rank is not a reduced case here, it is the
# whole domain of that file.
mftl_add_mpi_test(test_remeshing 1)

# The rim curve: where split, collapse and smoothing put a vertex of an open
# surface's rim, and the application's rim constraint. Serial fixtures run on
# every rank alone; at 2 and 4 a distributed disk keeps its rim on its circle
# where the rim crosses ranks (asserted), and ranks disagreeing about having a
# projector refuse to remesh.
mftl_add_mpi_test(test_rim_curves 1)
mftl_add_mpi_test_at(test_rim_curves_np2 test_rim_curves 2)
mftl_add_mpi_test_at(test_rim_curves_np4 test_rim_curves 4)

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
#       only the odd-parity phases resolve.
mftl_add_mpi_test(test_remeshing_parallel 1)
mftl_add_mpi_test_at(test_remeshing_parallel_np2 test_remeshing_parallel 2)
mftl_add_mpi_test_at(test_remeshing_parallel_np4 test_remeshing_parallel 4)
mftl_add_mpi_test_at(test_remeshing_parallel_np8 test_remeshing_parallel 8)
#
#   27  3x3x3. Two things only this grid can see, and they are different
#       questions. First, at cuts[k] == 2 a periodic axis has the SAME rank on
#       both sides of a cell, so the shifted ownership rule can confuse "one cell
#       left" with "one cell right" and no test at np <= 8 notices; three is the
#       smallest count where those are different ranks. Second, h falls to L/3,
#       so the h > 4*delta precondition stops being comfortable and the band
#       measurement is exercised against a real margin rather than a generous one.
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

# --- Coupling to a flow solver ------------------------------------------------
#
# The kernel table and the row core are local, so they run without MPI. The
# coupler is registered at the counts every distributed module uses, and each
# reaches something the others cannot: 1 is the serial reference path with
# every route SELF; 2 a thin 2x1x1 grid on which a periodic axis aliases both
# neighbours onto one rank; 4 the first diagonal neighbour; 8 every one of the
# 26 offsets populated; 27 an interior rank whose 26 neighbours are distinct.
# The solver's decomposition is varied INSIDE the test (aligned, z-slabs,
# reversed, all on rank 0), so every count also runs mismatched partitions.
mftl_add_test(test_coupling_kernels)
mftl_add_mpi_test(test_coupling 1)
mftl_add_mpi_test_at(test_coupling_np2 test_coupling 2)
mftl_add_mpi_test_at(test_coupling_np4 test_coupling 4)
mftl_add_mpi_test_at(test_coupling_np8 test_coupling 8)
mftl_add_mpi_test_oversub(test_coupling_np27 test_coupling 27)
# The contracts rather than the numbers: coverage, selection, staleness,
# failure, agreement. Several assertions only bite with a second rank (ranks
# that disagree, a duplicate owner on another rank), so 1 is not enough alone.
mftl_add_mpi_test(test_coupling_protocol 1)
mftl_add_mpi_test_at(test_coupling_protocol_np2 test_coupling_protocol 2)
mftl_add_mpi_test_at(test_coupling_protocol_np4 test_coupling_protocol 4)
mftl_add_mpi_test_at(test_coupling_protocol_np8 test_coupling_protocol 8)
# More ranks than point-cloud slabs also exercises empty provider regions.
mftl_add_mpi_test_oversub(test_coupling_protocol_np27 test_coupling_protocol 27)
# Configurations chosen to make the coupler return a plausible wrong answer:
# irregular multi-block decompositions with per-axis ghosts, anisotropic offset
# lattices, markers on the seam and on bin and rank boundaries, interleaved
# components, interleaved couplers. Against brute force over the global lattice.
mftl_add_mpi_test(test_coupling_adversarial 1)
mftl_add_mpi_test_at(test_coupling_adversarial_np2 test_coupling_adversarial 2)
mftl_add_mpi_test_at(test_coupling_adversarial_np4 test_coupling_adversarial 4)
mftl_add_mpi_test_at(test_coupling_adversarial_np8 test_coupling_adversarial 8)
mftl_add_mpi_test_oversub(test_coupling_adversarial_np27 test_coupling_adversarial 27)
# Who completes what: the solver-completed transpose of the ghost route (checked
# against a reverse halo accumulation the test performs itself), stage binds
# whose status the caller agrees (called by some ranks only: a collective inside
# would deadlock), and the later lifecycle contracts.
mftl_add_mpi_test(test_coupling_lifecycle 1)
mftl_add_mpi_test_at(test_coupling_lifecycle_np2 test_coupling_lifecycle 2)
mftl_add_mpi_test_at(test_coupling_lifecycle_np4 test_coupling_lifecycle 4)
mftl_add_mpi_test_at(test_coupling_lifecycle_np8 test_coupling_lifecycle 8)
mftl_add_mpi_test_oversub(test_coupling_lifecycle_np27 test_coupling_lifecycle 27)
# Surface quadrature: the points against the triangles, the level against the
# fluid, and the vertex <-> point conversions as an exact adjoint pair across
# the ghost schedule (compared with the same surface on one rank alone).
mftl_add_mpi_test(test_coupling_quadrature 1)
mftl_add_mpi_test_at(test_coupling_quadrature_np2 test_coupling_quadrature 2)
mftl_add_mpi_test_at(test_coupling_quadrature_np4 test_coupling_quadrature 4)
mftl_add_mpi_test_at(test_coupling_quadrature_np8 test_coupling_quadrature 8)
mftl_add_mpi_test_oversub(test_coupling_quadrature_np27 test_coupling_quadrature 27)
# Walls: the mirror boundary policy against a reference built from lattice
# INDICES (no reflection shared), the image rule's exactness, and the balance of
# force and power with the walls' reactions included.
mftl_add_mpi_test(test_coupling_boundaries 1)
mftl_add_mpi_test_at(test_coupling_boundaries_np2 test_coupling_boundaries 2)
mftl_add_mpi_test_at(test_coupling_boundaries_np4 test_coupling_boundaries 4)
mftl_add_mpi_test_at(test_coupling_boundaries_np8 test_coupling_boundaries 8)
mftl_add_mpi_test_oversub(test_coupling_boundaries_np27 test_coupling_boundaries 27)
# Anisotropic point locations: a rotated anisotropic lattice registered as a
# cloud with its cell matrix -- complete in the cell's frame, so Peskin needs no
# correction -- against brute force, and cells that rotate with position.
mftl_add_mpi_test(test_coupling_anisotropic 1)
mftl_add_mpi_test_at(test_coupling_anisotropic_np2 test_coupling_anisotropic 2)
mftl_add_mpi_test_at(test_coupling_anisotropic_np4 test_coupling_anisotropic 4)
mftl_add_mpi_test_at(test_coupling_anisotropic_np8 test_coupling_anisotropic 8)
mftl_add_mpi_test_oversub(test_coupling_anisotropic_np27 test_coupling_anisotropic 27)

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

# The same rule for topology changes: rasterisation, marching cubes, labelling
# and the id decision are local computation, and every exchange they need goes
# through topo_comm.c -- which is what lets the reconstruction be argued
# partition-independent one file at a time.
add_test(NAME test_topology_change_mpi_is_confined
         COMMAND ${CMAKE_COMMAND}
                 -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../src/topology_change
                 -DALLOWED=topo_comm.c
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_mpi_confinement.cmake)

# And the same rule for the coupler: discovery, requests, replies, the fetch
# and the spread schedules all go through coupling_comm.c, which never touches
# the solver's communicator.
add_test(NAME test_coupling_mpi_is_confined
         COMMAND ${CMAKE_COMMAND}
                 -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../src/coupling
                 -DALLOWED=coupling_comm.c
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

# The periodic extent rule. The measurement half is local and the same at any
# count; the check half is a per-body reduction like the one above, registered
# at the same counts for the same reasons, with its bodies redistributed so
# that the verdicts are genuinely combined over several holders (asserted).
mftl_add_mpi_test(test_periodic_extents 1)
mftl_add_mpi_test_at(test_periodic_extents_np2 test_periodic_extents 2)
mftl_add_mpi_test_at(test_periodic_extents_np4 test_periodic_extents 4)
mftl_add_mpi_test_at(test_periodic_extents_np8 test_periodic_extents 8)
mftl_add_mpi_test_oversub(test_periodic_extents_np27 test_periodic_extents 27)

# Per-body area, volume and centroid, closed and wall-closed, against
# references computed on rank 0 before partitioning by code that shares nothing
# with the library's: the same counts and reasons as the extents above.
mftl_add_mpi_test(test_body_integrals 1)
mftl_add_mpi_test_at(test_body_integrals_np2 test_body_integrals 2)
mftl_add_mpi_test_at(test_body_integrals_np4 test_body_integrals 4)
mftl_add_mpi_test_at(test_body_integrals_np8 test_body_integrals 8)
mftl_add_mpi_test_oversub(test_body_integrals_np27 test_body_integrals 27)

# Redistribution is the one module whose behaviour is a FUNCTION of the process
# grid, so it is registered at four counts and each is genuinely different:
#
#   1   the no-op path (and the guarantee that it really is a no-op)
#   2   a thin 2x1x1 grid: periodic wrap maps several of the 26 neighbour offsets
#       onto the same rank, so the neighbour list is shorter than the stencil
#   4   2x2x1: the first grid with a real diagonal neighbour
#   8   2x2x2: every one of the 26 offsets is populated (every cell is still a
#       26-neighbour of every other, so the scatter stays on the graph route;
#       the global route needs a cell two or more cells away, e.g. 3x3x3)
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
# The three share fixed output file prefixes, so under `ctest -j` they would
# write and delete each other's files.
set_tests_properties(test_io test_io_np2 test_io_np4 PROPERTIES RESOURCE_LOCK mftl_test_io_files)

# The settings module. Its text handling is rank-local and the same at any
# count; np = 2 is what tests the three collective promises -- only rank 0
# reads the file yet every rank gets the same struct and verdict, ranks holding
# DIFFERENT settings are caught before they can disagree about a collective,
# and a length derived from the mesh is bit-identical everywhere. Both write
# the same scratch file, hence the lock.
mftl_add_mpi_test(test_settings 1)
mftl_add_mpi_test_at(test_settings_np2 test_settings 2)
set_tests_properties(test_settings test_settings_np2 PROPERTIES RESOURCE_LOCK mftl_test_settings_files)

# MFTLComputeVertexNormals is a LOCAL operation, so a second rank count is not
# about communication -- it is about the empty-object path. MFTLGenerateSphere
# gives rank 0 the whole mesh, so at nproc == 3 the other two ranks run the pass
# on a pool with no allocation at all, and any collective accidentally introduced
# into it would deadlock here rather than in production.
mftl_add_mpi_test_at(test_vertex_normals_np3 test_vertex_normals 3)

# The scaling benchmark is an EXAMPLE, and no scaling sweep belongs in this
# suite -- the build guide (MFTL2_Documentation/pages/building.md) states the
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

  # The same flow driven entirely by a settings file, the way a solver would
  # integrate the library: every call takes its block from MFTL_Settings. The
  # overrides switch on the topology and contact blocks, so the lengths
  # MFTLSettingsResolve() derives for them reach a real call, and route the
  # arguments through MFTLSettingsLoadArgs() on a multi-rank run. The physical
  # check is the one test_leveque_parallel_conserves makes, for the same
  # reason: holding an MFTL_Object* across a redistribution breaks nothing
  # structural.
  add_test(NAME test_settings_driven_run_conserves
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:settings_driven_run>
                   ${CMAKE_CURRENT_SOURCE_DIR}/../examples/settings/settings_driven_run.toml
                   --steps 12 --quiet --verify topology.every=4 contact.every=4 contact.self_pairs=true
                   ${MPIEXEC_POSTFLAGS})

  # Every settings file shipped with the examples must stay loadable: a key
  # renamed in src/settings/settings_keys.c fails here, with the line, rather
  # than in somebody's job.
  file(GLOB _shipped_settings CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../examples/settings/*.toml")
  foreach(_file IN LISTS _shipped_settings)
    get_filename_component(_name "${_file}" NAME_WE)
    add_test(NAME test_shipped_settings_${_name}
             COMMAND $<TARGET_FILE:mftl_settings> check ${_file} --np 4)
  endforeach()

  # COUPLING, measured on a flow: LeVeque's vortex with the velocity known only
  # on a solver's samples -- a MAC grid owned in z-slabs (np = 2: every
  # marker's support crosses the solver's partition) and an unstructured
  # tetrahedral mesh owned in Morton chunks (np = 4) -- advected with RK4
  # through stage binds. -verify fails the run unless the vertex error at
  # maximum deformation converges at better than order 1.5 between the two
  # meshes, and the remeshed front-tracking run stays a closed sphere that
  # tracks the direct run's volume. Half a period: the error at T is no
  # coupling measure (the interpolated field is time-reversible too).
  add_test(NAME test_leveque_coupled_cartesian
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:leveque_coupled_cartesian> -n 16,32 -steps 75 -verify ${MPIEXEC_POSTFLAGS})
  add_test(NAME test_leveque_coupled_unstructured
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:leveque_coupled_unstructured> -n 8,16 -steps 75 -verify ${MPIEXEC_POSTFLAGS})

  # OPEN SURFACES. No library test runs the distributed remesher on a surface
  # with a rim; these four examples do, and each checks what an open surface
  # must keep that a closed one cannot lose: every twin-less half-edge of an
  # owned triangle carries MFTL_FLAG_BOUNDARY (a short halo otherwise), the Euler
  # characteristic of a disk or an annulus, every rim vertex on its constraint,
  # no triangle turned over against all its neighbours, and a conserved volume.
  #
  # --require-rim-cut is the non-vacuity measure: each run fails unless some rim
  # triangle was held by two ranks at once. Without it a grid or resolution change
  # can leave the rim inside one rank's box, and the test would keep passing
  # with nothing about parallel rims exercised.
  #
  # open_surface_creation   no remeshing: a patch and a cap crossing a periodic
  #                         2x2x1 grid, area/entities/rim exact to round-off
  # rayleigh_taylor         a rim on four walls, cut by both cut planes of the
  #                         default 2x2x1 grid
  # travelling_waves        a rim on two walls AND a periodic seam, i.e. an
  #                         annulus that must stay one; at np = 2 the seam is
  #                         itself a partition boundary
  # liquid_jet              a surface growing through a pinned inflow rim, on
  #                         2x2x1 so that a cut plane runs through the jet axis
  #                         and the lip; the default (P,1,1) grid would leave the
  #                         lip inside rank 0's box
  add_test(NAME test_open_surface_creation_conserves
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:open_surface_creation>
                   2 2 1 --steps 20 --no-output --require-rim-cut ${MPIEXEC_POSTFLAGS})
  add_test(NAME test_rayleigh_taylor_open_remesh
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:rayleigh_taylor_open_surface>
                   --res 24 --steps 60 --vtk 0 --report 20 --require-rim-cut ${MPIEXEC_POSTFLAGS})
  add_test(NAME test_travelling_waves_open_remesh
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:travelling_waves_open_surface>
                   --nx 64 --ny 16 --steps 100 --vtk 0 --report 25 --require-rim-cut ${MPIEXEC_POSTFLAGS})
  add_test(NAME test_liquid_jet_open_remesh
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:liquid_jet_open_surface>
                   2 2 1 --sectors 24 --steps 60 --vtk 0 --report 20 --require-rim-cut ${MPIEXEC_POSTFLAGS})

  # TOPOLOGY CHANGES on the same three open surfaces. Each run must make a
  # change actually happen -- the non-vacuity flag fails it otherwise -- and
  # keep every structural and physical check of its plain run through it, with
  # the Euler rule generalised per body and the volume the reconstructions moved
  # booked separately:
  #
  #   liquid_jet   a neck pinches and a drop travels on as a closed body to the
  #                outlet; 2x2x1 so the lip, the jet and the neck are all cut
  #   rayleigh_    a stronger vortex ring rolls the cap until turns touch and
  #   taylor       a blob of heavy fluid pinches off it (t ~ 3.5), surviving the
  #                remesh as a closed body. A coarser patch pinches only blobs
  #                too small to resolve, which the size floor deletes: an
  #                honest run in which nothing visible changes
  #   travelling_  a steep carrier with a ripple plunges and its lip reconnects
  #   waves        with the face, trapping air as an open body between the
  #                channel walls. 104 steps: past that the prescribed field's
  #                e^{kz} feedback runs away, which the example's own guard
  #                reports and which is not the reconstruction's to fix.
  #                96 x 24, not 64 x 16: at 64 x 16 the plunging lip is three
  #                edges thick and whether it reconnects, and whether the
  #                post-reconstruction remesh leaves a transient fold at a wall
  #                (a remesher weakness at creases, docs/analysis.md §9), is
  #                chaotic -- 4 of 20 neighbouring (steepness, ripple) points
  #                passed on the code this test was tuned on, this one among
  #                them, and 1 of 20 once the rim curve changed the trajectory.
  #                At 96 x 24: 6 and 8 of 20, this point passing on both
  add_test(NAME test_liquid_jet_topology
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:liquid_jet_open_surface>
                   2 2 1 --sectors 24 --length 24 --eps 0.1 --weber 2 --vtk 0 --report 20
                   --require-breakup --require-rim-cut ${MPIEXEC_POSTFLAGS})
  add_test(NAME test_rayleigh_taylor_topology
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:rayleigh_taylor_open_surface>
                   --roll 1.6 --vtk 0 --report 50
                   --require-topology --require-rim-cut ${MPIEXEC_POSTFLAGS})
  add_test(NAME test_travelling_waves_topology
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:travelling_waves_open_surface>
                   --nx 96 --ny 24 --components 2 --steepness 0.55 --ripple 0.05 --delta-breakup 0.02
                   --steps 104 --vtk 0 --report 8 --require-topology --require-rim-cut ${MPIEXEC_POSTFLAGS})
endif()
