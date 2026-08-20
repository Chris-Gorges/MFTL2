# The Doxygen manual

The complete Doxygen manual is published as part of this site, at
[`/doxygen/`](../doxygen/index.html){ target="_blank" }. It is not a
duplicate of the [API reference](index.md): it carries three classes of artefact
that come out of Graphviz and out of Doxygen's source parser, and that no
Markdown rendering of the XML can reproduce.

<div class="grid cards" markdown>

-   :material-graph-outline:{ .lg .middle } &nbsp; __Call and caller graphs__

    ---

    For every function, both directions. The interesting ones &mdash;
    `MFTLRemeshEnsembleParallel`, `MFTLRedistributeListOfObjects` &mdash; are
    emitted as interactive SVG, so they pan and zoom rather than merely
    existing.

    [:octicons-arrow-right-24: Function index](../doxygen/globals_func.html){ target="_blank" }

-   :material-file-code-outline:{ .lg .middle } &nbsp; __Annotated sources__

    ---

    The full source browser, cross-linked in both directions, with comments
    preserved. The prose in this library lives next to the code, so the
    annotated sources are a first-class part of the manual rather than a dump.

    [:octicons-arrow-right-24: Source files](../doxygen/files.html){ target="_blank" }

-   :material-sitemap-outline:{ .lg .middle } &nbsp; __Include dependency graphs__

    ---

    Which header includes which, drawn. This is the fastest way to check the
    layering rule &mdash; a header may only include headers from a strictly
    lower layer &mdash; because a violation shows up as an edge pointing the
    wrong way.

    [:octicons-arrow-right-24: Topics by layer](../doxygen/topics.html){ target="_blank" }

-   :material-magnify:{ .lg .middle } &nbsp; __A second search index__

    ---

    Doxygen's own client-side search, which indexes members and their
    signatures rather than page text. Useful when you half-remember a
    parameter name.

    [:octicons-arrow-right-24: Open the manual](../doxygen/index.html){ target="_blank" }

</div>

!!! note "Rebuilding it"

    The manual is copied into this site from `build/docs/html`, which is what
    `python3 build_docs.py` produces. `python3 build_mkdocs.py` runs both steps
    in the right order. If the links above 404, the Doxygen pass has not run in
    this tree yet:

    ``` bash
    python3 build_docs.py          # -> build/docs/html
    python3 build_mkdocs.py        # -> build/site, with /doxygen/ inside it
    ```

!!! warning "The zero-warning criterion"

    `WARN_IF_UNDOCUMENTED`, `WARN_IF_INCOMPLETE_DOC`, `WARN_NO_PARAMDOC` and
    `WARN_IF_UNDOC_ENUM_VAL` are all on, and the manual currently builds with
    **zero** warnings. That is the acceptance criterion, so a new function
    without `@param`/`@return` fails `python3 build_docs.py --strict`. Run it
    after touching any header.
