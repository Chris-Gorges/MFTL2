"""Read Doxygen's ``index.xml`` so cross-references can be resolved exactly.

mkdoxy generates one Markdown page per Doxygen *compound* and names it after the
compound's ``refid``.  The same ``refid`` is in ``index.xml``, so parsing that
one file is enough to turn ``@ref mftl_object``, ``#MFTL_Object`` and
``MFTLObjectCreate()`` into links that are guaranteed to hit a page mkdoxy
actually wrote -- rather than a path guessed from Doxygen's name-mangling rules,
which is where this kind of integration normally rots.

Only ``index.xml`` is read.  It carries every compound and every member with its
kind and its owning compound, which is exactly what :class:`doxymd.LinkIndex`
needs and nothing more.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

#: Compound kinds worth linking to.  Namespaces and directories are omitted:
#: this is C, so there are no namespaces, and a directory page tells a reader
#: nothing the module theory pages do not say better.
_LINKABLE_COMPOUNDS = {"file", "struct", "union", "group", "page"}

#: Member kinds mkdoxy documents under a ``### <kind> <name>`` heading.  An
#: ``enumvalue`` is deliberately absent: it is rendered inside its enum's table
#: rather than as a heading, so it resolves to the page without a fragment.
_ANCHORED_MEMBERS = {"function", "define", "typedef", "variable", "enum"}


def _compound_rank(name: str) -> int:
    """Rank a compound as a home for a member, lowest wins.

    A function is declared in a public header and defined in a translation
    unit, so Doxygen lists it twice.  The header is the better landing page for
    a reader following a cross-reference: it is the file they will ``#include``.

    :param name: the compound's name, e.g. ``mftl_object.h`` or ``object.c``.
    :returns: sort key, 0 for a public MFTL header, 3 for anything else.
    """
    if name.startswith("mftl_") and name.endswith(".h"):
        return 0
    if name.endswith(".h"):
        return 1
    if name.endswith(".c"):
        return 2
    return 3


def load_index(xml_dir: str) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    """Parse ``<xml_dir>/index.xml``.

    :param xml_dir: directory holding Doxygen's XML output.
    :returns: ``(compounds, members)`` where ``compounds`` maps a compound name
        to ``(refid, kind)`` and ``members`` maps a member name to
        ``(owning compound refid, kind)``.  Both are empty when the XML has not
        been generated yet, which lets the page generator degrade to plain code
        spans instead of failing the build.
    """
    index_path = os.path.join(xml_dir, "index.xml")
    if not os.path.isfile(index_path):
        return {}, {}

    compounds: dict[str, tuple[str, str]] = {}
    members: dict[str, tuple[str, str]] = {}
    owner_rank: dict[str, int] = {}

    root = ET.parse(index_path).getroot()
    for compound in root.findall("compound"):
        kind = compound.get("kind", "")
        refid = compound.get("refid", "")
        name_el = compound.find("name")
        if name_el is None or name_el.text is None:
            continue
        name = name_el.text.strip()

        if kind in _LINKABLE_COMPOUNDS and name not in compounds:
            compounds[name] = (refid, kind)

        if kind not in ("file", "struct", "union", "group"):
            continue
        rank = _compound_rank(name)
        for member in compound.findall("member"):
            member_kind = member.get("kind", "")
            member_name_el = member.find("name")
            if member_name_el is None or member_name_el.text is None:
                continue
            member_name = member_name_el.text.strip()
            if not member_name or member_kind not in _ANCHORED_MEMBERS | {"enumvalue"}:
                continue
            if member_name in members and owner_rank.get(member_name, 9) <= rank:
                continue
            members[member_name] = (refid, member_kind if member_kind in _ANCHORED_MEMBERS else "")
            owner_rank[member_name] = rank

    return compounds, members


def find_xml_dir(candidates: list[str]) -> str | None:
    """Return the first candidate directory that holds an ``index.xml``.

    :param candidates: directories to try, most specific first.
    :returns: the directory, or ``None`` when Doxygen has not run yet.
    """
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "index.xml")):
            return candidate
    return None
