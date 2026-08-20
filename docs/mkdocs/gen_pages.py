"""Generate the manual's prose pages from the repository's own Markdown.

Run by ``mkdocs-gen-files`` during every build (and every ``mkdocs serve``
reload), this script is the single seam between the two documentation
backends.  It reads the files Doxygen reads -- ``docs/pages/*.md``,
``src/<module>/README.md``, ``docs/*-analysis.md`` -- and writes the same prose
into the MkDocs site as virtual pages, rewritten by :mod:`doxymd`.

Nothing is copied on disk.  There is one source of truth per subject, and both
manuals are built from it, which is the same rule ``CLAUDE.md`` states for the
Doxyfile template.

Cross-references are resolved in two passes because the corpus is a graph: the
first pass records every ``{#label}`` and where it will live, the second
converts.  A reference into the API reference is resolved against Doxygen's
``index.xml`` (see :mod:`doxyxml`), so it can only ever point at a page mkdoxy
has generated.
"""

from __future__ import annotations

import os
import sys

import mkdocs_gen_files

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import doxymd  # noqa: E402
import doxyxml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: ``(source file, site page)`` for the guide pages, in nav order.
GUIDE_PAGES: list[tuple[str, str]] = [
    ("docs/pages/mainpage.md", "overview.md"),
    ("docs/pages/architecture.md", "guide/architecture.md"),
    ("docs/pages/building.md", "guide/building.md"),
    ("docs/pages/timestep.md", "guide/timestep.md"),
    ("docs/pages/invariants.md", "guide/invariants.md"),
    ("docs/pages/modules.md", "modules/index.md"),
    ("docs/pages/analysis.md", "analysis/index.md"),
]

#: The ten module theory pages, in the layer order ``docs/pages/modules.md``
#: presents them.  Each is one ``src/<dir>/README.md``.
MODULE_DIRS: list[str] = [
    "arrayfunctions",
    "object",
    "topology",
    "geometry",
    "mesh_creation",
    "ensemble",
    "communication",
    "remeshing",
    "bvh",
    "io",
]

#: The three measurement documents.
ANALYSIS_PAGES: list[tuple[str, str]] = [
    ("docs/performance-analysis.md", "analysis/performance.md"),
    ("docs/remeshing-quality-analysis.md", "analysis/remeshing-quality.md"),
    ("docs/parallel-remeshing-analysis.md", "analysis/parallel-remeshing.md"),
]

#: Where mkdoxy's Doxygen XML may be found, most likely location first.  The
#: plugin writes it under ``save-api`` when that is configured and under the
#: site directory otherwise; a standalone ``build_docs.py`` run leaves a third
#: copy.  Any of them answers the same questions.
XML_CANDIDATES = [
    os.path.join(ROOT, "build", "mkdocs", ".doxy", "mftl", "xml"),
    os.path.join(ROOT, "build", "site", "assets", ".doxy", "mftl", "xml"),
    os.path.join(ROOT, "build", "docs", "xml"),
]


def _source_note(source: str) -> str:
    """Footer marking a page as generated, and naming the file to edit instead.

    :param source: repository-relative path of the source document.
    :returns: a Markdown fragment.
    """
    return (
        '\n\n!!! abstract "Where this page comes from"\n\n'
        f"    This page is generated from `{source}` in the repository, which is also\n"
        "    what the Doxygen manual reads. Edit that file, not this page.\n"
    )


def _pages() -> list[tuple[str, str]]:
    """Assemble the full ``(source, site page)`` list.

    :returns: every page this script generates, in a stable order.
    """
    pages = list(GUIDE_PAGES)
    pages += [(f"src/{d}/README.md", f"modules/{d}.md") for d in MODULE_DIRS]
    pages += ANALYSIS_PAGES
    return pages


def main() -> None:
    """Harvest anchors, then convert and emit every page.

    :returns: nothing.
    """
    index = doxymd.LinkIndex(api_path="api")

    xml_dir = doxyxml.find_xml_dir(XML_CANDIDATES)
    if xml_dir:
        index.compounds, index.members = doxyxml.load_index(xml_dir)

    pages = _pages()
    sources: dict[str, str] = {}
    for source, page in pages:
        path = os.path.join(ROOT, source)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            sources[source] = handle.read()
        doxymd.harvest_anchors(sources[source], page, index)

    for source, page in pages:
        if source not in sources:
            continue
        body = doxymd.convert(sources[source], page, index)
        with mkdocs_gen_files.open(page, "w") as handle:
            handle.write(body)
            handle.write(_source_note(source))
        mkdocs_gen_files.set_edit_path(page, source)


main()
