The per-timestep loop {#mftl_page_timestep}
=====================

[TOC]

This page is the call-order contract. Everything on it is load-bearing, and most
of it is **silent when wrong** — the mesh stays structurally valid, every
integrity check still passes, and the numbers are quietly stale or quietly
partial.

---

# Without redistribution {#ts_no_redist}

When the partition is not being changed this step:

```c
MFTL_FOREACH_OWNED_VERTEX(obj, /* move p[0..2] */ );   /* 1. advect         */
MFTLObjectRefreshFrames(obj);                          /* 2. rebuild frames */
MFTLEnsembleUpdate(ens);                               /* 3. geometry+stats */
MFTLComputeVertexNormals(obj);                         /* 4. optional       */
```

#MFTL_FOREACH_OWNED_VERTEX clears #MFTL_DERIVED_FRAMES on the way out, so step 2
cannot be forgotten without a Debug assertion firing in step 3.

Step 4 is *local* and needs no communication, but only because the halo is
vertex-complete — see @ref ts_vertex_normals below.

---

# With redistribution {#ts_redist}

```c
/* 1 */ MFTL_FOREACH_OWNED_VERTEX(obj, /* move p[0..2] */ );
/* 2 */ MFTLUpdateGhostPositions(ens);
/* 3 */ MFTLComputeTriangleGeometry(obj, obj->tris.n);      /* note: n, not n_local */
/* 4 */ MFTLRedistribute(ens);
/* 5 */ MFTLEnsembleUpdate(ens);              /* optional, but see below */
/* 6 */ MFTLComputeVertexNormals(obj);        /* optional */
```

Three things about this order are load-bearing, and **all three are silent when
wrong**:

### The ghost sync must precede the geometry pass

A triangle's three vertices can belong to three different ranks. Computing its
centroid before step 2 uses last step's positions for whichever corners are
ghosts.

### The geometry pass must cover the ghosts too

Hence `obj->tris.n` and not `obj->tris.n_local` in step 3. Redistribution routes
triangles by their **cached centre**; refreshing only the owned ones makes two
ranks disagree about a ghost's destination, and the entity is duplicated or lost
depending on which way the disagreement falls.

### There is deliberately no MFTLObjectRefreshFrames() in the loop

MFTLComputeTriangleGeometry() reads the half-edges directly, and the
redistribution rebuilds the frames on its way out. Anything wanting the frames
goes **after** step 4.

---

# What a redistribution leaves behind {#ts_redist_leaves}

@warning **Redistribution invalidates all cached geometry except the vertex
         positions.**

It returns an object with valid topology, valid positions and **both** derived
bits set. But:

| Array | State after MFTLRedistribute() |
|---|---|
| `verts.pos` | valid |
| `tris.verts`, `tris.neighbors`, `tris.anchor`, `tris.edge` | valid (rebuilt) |
| `tris.area`, `tris.normal`, `tris.center` | **unspecified** |
| `verts.normal`, `verts.area`, `verts.mean_curv`, `verts.gauss_curv` | **unspecified** |

That is what buys the 3× smaller wire format: the alternative is to pack the
areas, normals and centres onto the wire so that a timestep ends with valid
triangle geometry. A solver that consumes triangle geometry *after* the
redistribution therefore owes one more pass.

`examples/parallel_advection_redistribute_list.c` runs it as step 5 and times it
**separately**, because a redistribute timing that excludes the pass it defers
credits the redistribution with a saving it has not finished paying for.

@note Step 5 is bound by **first touch**, not by its kernel. The geometry pass
      that restores what redistribution drops runs on pools the redistribution
      has just rewritten, and costs 6.4 ns/triangle against 2.9 ns for the same
      function on a resident mesh.

---

# Vertex normals and the vertex-complete halo {#ts_vertex_normals}

MFTLComputeVertexNormals() accumulates

\f[
  \vect{N}_v = \frac{\sum_{t \ni v} A_t\, \hat{\vect{n}}_t}
                    {\norm{\sum_{t \ni v} A_t\, \hat{\vect{n}}_t}}
\f]

over **all** triangles including ghosts. An owned vertex therefore gets the exact
answer *iff* every triangle incident to it is present locally.

That is a stronger requirement on redistribution than "one layer of edge-adjacent
ghost triangles", and it is exactly what buys the pass zero communication. The
redistribution halo rule — a triangle travels to the owner of each of its three
**corners** as well as each of its three edge neighbours — is what supplies it.

@warning **Ghost vertices hold a PARTIAL SUM**, correctly normalised and quietly
         wrong. That is the trap: it looks exactly like a real normal.
         MFTLEnsembleUpdateVertexNormals() closes it with one ghost exchange.
         Call that unless you are certain you read owned vertices only.

---

# Which calls are collective {#ts_collective}

**Collective** — every rank of the ensemble's communicator must call them:

- MFTLEnsembleCreate(), MFTLEnsembleUpdate(), MFTLEnsembleUpdateGlobalCounts()
- MFTLGenerateSphere(), MFTLBuildIndexedMesh()
- MFTLRedistribute(), MFTLRedistributeObject(), MFTLRedistributeListOfObjects(),
  MFTLRedistributeWithOwners()
- MFTLUpdateGhostPositions(), MFTLUpdateGhostVertexNormals(),
  MFTLUpdateGhostGlobalIds()
- MFTLObjectReorderMorton(), MFTLObjectUpdateGlobalStats()
- MFTLEnsembleUpdateVertexNormals(), MFTLEnsembleMeanEdgeLength()
- MFTLWriteVtu()
- MFTLRemeshEnsemble(), MFTLRemeshEnsembleParallel(),
  MFTLRemeshEnsembleDispersed(), MFTLRemeshChooseStrategy(),
  MFTLRemeshMeasureBodies()
- MFTLExtendSpatialHalo(), MFTLSpatialHaloResolveRadii(),
  MFTLEnsembleIntersectObjects()

**Deliberately local** — each rank works on its own partition, and calling them
from a subset of ranks is fine:

- MFTLObjectBuildDerived(), MFTLObjectRefreshFrames()
- MFTLComputeTriangleGeometry(), MFTLComputeVertexNormals()
- MFTLRemeshObject(), MFTLRemeshMarkFrozen(), MFTLDilateFrozenBand(),
  MFTLDetectFeatures(), MFTLSmoothTangential(), MFTLMeanEdgeLength()
- MFTLWriteVtk()
- MFTL_GenerateLBVH(), MFTL_RefitLBVH(), MFTL_IntersectObjects()

`test_vertex_normals_np3` exists to make an accidentally-introduced collective in
the local list deadlock in CI.

@warning MFTLRemeshEnsemble() **dispatches**: `nproc == 1` takes the serial path
         unchanged; above that MFTLRemeshChooseStrategy() (itself collective — it
         measures the bodies) picks the dispersed or the spanning driver. **Both
         drivers must be reached by every rank**, so the choice is made from
         reduced data and never from anything local. Two ranks entering different
         drivers is an immediate deadlock with no diagnostic.

---

# Adding remeshing to the loop {#ts_remeshing}

Remeshing is not a per-timestep necessity for every solver, but when it is in the
loop the ordering constraint is:

```
advect  →  ghost sync  →  geometry  →  REDISTRIBUTE  →  REMESH  →  geometry again
```

Remeshing must follow the redistribution, not precede it, because the parallel
remesher's freeze test reads the *current* partition: an entity is frozen exactly
when some other rank holds a copy of it, and MFTLRemeshMarkFrozen() derives that
from `corner >= verts.n_local || neighbour >= tris.n_local`.

@warning `MFTL_RemeshParameters::parallel_phase_budget` defaults to 32, which
         means *run until no rank wants to do anything*. That is right for a
         one-shot remesh and measurably wrong inside a per-timestep loop, in both
         directions at once — worse volume conservation, worse triangle quality
         **and** five times slower. Set it to 2–8 for a loop. The full
         measurement table is on the field's documentation.

@note Remeshing renumbers global ids at the end of every pass, and it has to:
      splits hand out fresh ids and collapses retire them, so ids stay unique but
      stop being dense — and `src/communication/` sizes its arrival lookup tables
      from `next_*_gid`. The cost is that pool indices **and** global ids are both
      unstable across a remesh, so nothing external may hold either.

---

# Adding contact detection to the loop {#ts_contact}

```
... → REDISTRIBUTE → MFTLExtendSpatialHalo() → MFTLEnsembleIntersectObjects()
```

@warning **The spatial halo is transient and cannot be otherwise.**
         Redistribution keeps only owned triangles, so a halo cannot survive a
         repartition even in principle. Each MFTLExtendSpatialHalo() *replaces*
         the previous patches rather than accumulating; grow it after
         redistribution and before contact detection.

@note **Contact detection is discrete-time: there is no CCD.** A pair whose
      relative displacement in one step exceeds the sum of their sizes passes
      through without ever being coincident at a sampled instant, and both
      samples correctly report nothing. This bites when the timestep is set by a
      CFL condition on the *flow* rather than by the approach velocity of two
      surfaces, which is the normal case.
