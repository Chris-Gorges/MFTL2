# API reference

Everything below is generated from the sources by
[Doxygen](https://www.doxygen.nl) and rendered into this site by
[mkdoxy](https://mkdoxy.kubaandrysek.cz), using the same
`docs/doxygen/Doxyfile.in` the standalone Doxygen manual is built from. The two
therefore extract the same symbols with the same aliases and the same
preprocessor settings; there is no second configuration to keep in step.

<div class="grid cards" markdown>

-   :material-file-tree:{ .lg .middle } &nbsp; __Topics__

    ---

    The library grouped by architectural layer &mdash; layer 0 foundation up to
    the module APIs at layer 3. The group hierarchy is defined in exactly one
    place, `docs/doxygen/groups.dox`.

    [:octicons-arrow-right-24: Browse by layer](modules.md)

-   :material-code-braces:{ .lg .middle } &nbsp; __Data structures__

    ---

    Every struct. The public ones are all defined in `mftl_types.h`; the rest
    are module-local.

    [:octicons-arrow-right-24: All structs](annotated.md) &middot;
    [fields](class_members.md)

-   :material-file-document-multiple-outline:{ .lg .middle } &nbsp; __Source files__

    ---

    Headers and translation units, each with its brief, its members and a link
    to the annotated source.

    [:octicons-arrow-right-24: All files](files.md)

-   :material-function-variant:{ .lg .middle } &nbsp; __Symbols__

    ---

    The alphabetical indexes.

    [:octicons-arrow-right-24: Functions](functions.md) &middot;
    [macros](macros.md) &middot;
    [variables](variables.md)

</div>

## Which manual answers which question

This site and the embedded Doxygen manual are generated from the same sources
but they are not the same artefact, and the split is by *what the tool can
produce*, not by taste.

| Question | Where |
|---|---|
| What does this function do, what are its parameters | here, or the Doxygen manual &mdash; identical content |
| What is the theory behind this module | here: the [module pages](../modules/index.md) |
| Who calls this function, and what does it call | the Doxygen manual &mdash; Graphviz call and caller graphs |
| What does the annotated source look like | the Doxygen manual &mdash; cross-linked source browser |
| Which headers include which | the Doxygen manual &mdash; `#include` dependency graphs |

[Open the Doxygen manual](doxygen-manual.md){ .md-button }

## Naming conventions the reference assumes

You will read the index faster knowing three rules the library follows without
exception:

- `MFTLPascalCase` is a **function**, `MFTL_PascalCase` is a **type**,
  `MFTL_UPPER_CASE` is a **macro or constant**. `snake_case` is a file-static
  helper, and it appears here only because `EXTRACT_STATIC` is on &mdash; the
  real algorithms live in those.
- A function returns `int`, either `MFTL_SUCCESS` or `MFTL_ERROR`. The same
  `MFTL_ERROR` (&minus;1) doubles as the "no such entity" index sentinel.
- Anything named `MFTLEnsemble*` is **collective** and must be called by every
  rank. Anything taking a single `MFTL_Object *` is rank-local unless its
  documentation says otherwise &mdash; and the ones that break that pattern
  carry an explicit *Collective* paragraph.
