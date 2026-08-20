"""MkDocs hooks that bind the Doxygen manual into the Material site.

Two manuals are generated from this repository and they answer different
questions.  The Material site is the *manual*: prose, guides, module theory and
a browsable API index.  The Doxygen manual is the *machine-generated reference*:
call graphs, caller graphs, ``#include`` dependency graphs and a fully
cross-linked source browser -- artefacts mkdoxy does not reproduce because they
come out of Graphviz, not out of the XML.

Rather than ask the reader to build and open a second site, the whole Doxygen
HTML tree is copied into the published site under ``/doxygen/``.  One deploy
target, one URL, both manuals.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import doxyfile  # noqa: E402

log = logging.getLogger("mkdocs.hooks.mftl")

#: Where ``build_docs.py`` (and the ``docs`` CMake target) leave the manual.
DOXYGEN_HTML = os.path.join("build", "docs", "html")

#: Where it is mounted inside the published site.
MOUNT_POINT = "doxygen"

#: Doxyfile handed to mkdoxy, generated from ``docs/doxygen/Doxyfile.in``.
API_DOXYFILE = os.path.join("build", "mkdocs", "Doxyfile.api")


def on_config(config, **kwargs):
    """Derive mkdoxy's Doxygen configuration from the project's own template.

    Runs before any plugin's ``on_files``, which is when mkdoxy reads the file
    named by its ``doxy-cfg-file`` option -- so the API reference is always
    generated with the same aliases, the same ``PREDEFINED`` set and the same
    extraction policy as the Doxygen manual, from one template.

    :param config: the MkDocs configuration.
    :param kwargs: further MkDocs event arguments, unused.
    :returns: the configuration, unchanged.
    """
    root = os.path.dirname(os.path.abspath(config["config_file_path"]))
    try:
        doxyfile.write_api_doxyfile(root, os.path.join(root, API_DOXYFILE))
    except OSError as error:
        log.warning("MFTL: could not generate %s (%s); mkdoxy will fall back to its defaults.", API_DOXYFILE, error)
    return config


def on_post_build(config, **kwargs) -> None:
    """Copy the Doxygen HTML manual into the built site.

    Silent about a missing manual beyond one informational line: a prose-only
    rebuild is a legitimate thing to do while editing, and failing it would make
    the fast path depend on Graphviz.

    :param config: the MkDocs configuration.
    :param kwargs: further MkDocs event arguments, unused.
    :returns: nothing.
    """
    root = os.path.dirname(os.path.abspath(config["config_file_path"]))
    source = os.path.join(root, DOXYGEN_HTML)
    target = os.path.join(config["site_dir"], MOUNT_POINT)

    if not os.path.isdir(source):
        log.info(
            "MFTL: no Doxygen manual at %s -- the '%s/' section of the site will 404. "
            "Run `python3 build_docs.py` (or `python3 build_mkdocs.py`, which does it for you).",
            DOXYGEN_HTML,
            MOUNT_POINT,
        )
        return

    if os.path.isdir(target):
        shutil.rmtree(target)
    shutil.copytree(source, target)
    log.info("MFTL: embedded the Doxygen manual at /%s/ (%d files).", MOUNT_POINT, sum(len(f) for _, _, f in os.walk(target)))
