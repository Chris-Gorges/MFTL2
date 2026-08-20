---
hide:
  - navigation
  - toc
---

# MFTL

<p class="mftl-hero-tagline">
A distributed-memory C11 library for triangulated surface meshes &mdash;
built to carry a front through a flow solver.
</p>

[What MFTL is](overview.md){ .md-button .md-button--primary }
[Build it](guide/building.md){ .md-button }
[API reference](api/index.md){ .md-button }
{ .mftl-hero-actions }

---

## Start here

<div class="grid cards" markdown>

-   :material-vector-triangle:{ .lg .middle } &nbsp; __Overview__

    ---

    What the library does today, what is not ported yet, and the shortest
    complete program that creates a sphere and writes it out.

    [:octicons-arrow-right-24: Read the overview](overview.md)

-   :material-hammer-wrench:{ .lg .middle } &nbsp; __Building &amp; testing__

    ---

    CMake, the three build modes, the forty CTest entries, and why
    `-march=native` is off by default.

    [:octicons-arrow-right-24: Build and run the suite](guide/building.md)

-   :material-layers-triple-outline:{ .lg .middle } &nbsp; __Architecture__

    ---

    One idea decides the file layout: topology wants AoS, geometry wants SoA,
    and both live in one allocation per entity class.

    [:octicons-arrow-right-24: Read the one idea](guide/architecture.md)

-   :material-timer-outline:{ .lg .middle } &nbsp; __The timestep__

    ---

    The per-timestep call order, with and without redistribution &mdash; and the
    three things in it that are silent when wrong.

    [:octicons-arrow-right-24: Get the order right](guide/timestep.md)

</div>

---

## Go deeper

<div class="grid cards" markdown>

-   :material-shield-check-outline:{ .lg .middle } &nbsp; __Invariants__

    ---

    The properties that must survive any change, each named together with the
    test that guards it.

    [:octicons-arrow-right-24: What must stay true](guide/invariants.md)

-   :material-view-module-outline:{ .lg .middle } &nbsp; __Ten modules__

    ---

    One module is one `src/` directory, one public header and one theory
    chapter. Start from the dependency graph.

    [:octicons-arrow-right-24: Browse the modules](modules/index.md)

-   :material-chart-line:{ .lg .middle } &nbsp; __Measured, not assumed__

    ---

    Three measurement documents, and the results that contradicted what the
    design predicted.

    [:octicons-arrow-right-24: Read the numbers](analysis/index.md)

-   :material-api:{ .lg .middle } &nbsp; __API reference__

    ---

    Every function, macro, struct and field, generated from the sources &mdash;
    plus the full Doxygen manual with call graphs.

    [:octicons-arrow-right-24: Open the reference](api/index.md)

</div>

---

## The shape of a run

``` mermaid
graph LR
  A[MFTLEnsembleCreate] --> B[MFTLGenerateSphere]
  B --> C{per timestep}
  C -->|move owned vertices| D[MFTLUpdateGhostPositions]
  D --> E[MFTLComputeTriangleGeometry]
  E --> F[MFTLRedistribute]
  F --> G[MFTLEnsembleUpdate]
  G --> H[MFTLRemeshEnsemble]
  H --> I[MFTLExtendSpatialHalo]
  I --> J[MFTL_IntersectObjects]
  J --> K[MFTLWriteVtu]
  K --> C
```

The order is not decorative. Three of those edges are load-bearing and fail
silently when swapped &mdash; [the timestep page](guide/timestep.md) says which
three and why.
