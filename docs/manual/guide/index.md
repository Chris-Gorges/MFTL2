# Guide

Four pages, in the order they are worth reading.

<div class="grid cards" markdown>

-   :material-hammer-wrench:{ .lg .middle } &nbsp; __Building &amp; testing__

    ---

    Requirements, the three build modes and what each is for, the CTest suite,
    running at several ranks, and how to regenerate this manual.

    [:octicons-arrow-right-24: guide/building](building.md)

-   :material-layers-triple-outline:{ .lg .middle } &nbsp; __Architecture__

    ---

    The one idea, the three invariants every module may assume, the derived
    arrays and their staleness bits, header layering, and the object model.

    [:octicons-arrow-right-24: guide/architecture](architecture.md)

-   :material-timer-outline:{ .lg .middle } &nbsp; __The timestep__

    ---

    The call order with and without redistribution, what redistribution leaves
    undefined, and where remeshing and contact detection fit.

    [:octicons-arrow-right-24: guide/timestep](timestep.md)

-   :material-shield-check-outline:{ .lg .middle } &nbsp; __Invariants__

    ---

    Every property that must survive a change, with the test that guards it and
    the symptom you get when it stops holding.

    [:octicons-arrow-right-24: guide/invariants](invariants.md)

</div>

!!! tip "Reading order"

    If you are here to *use* the library, **Building** then **The timestep** is
    enough to get a solver running. If you are here to *change* it, read
    **Architecture** first &mdash; most of the file layout follows from a single
    decision, and **Invariants** is the list of things that are silent when
    broken.
