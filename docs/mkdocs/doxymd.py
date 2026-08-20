"""Doxygen-flavoured Markdown to MkDocs-Material Markdown.

The prose of this library lives in exactly one place per subject:
``docs/pages/*.md`` for the guides, ``src/<module>/README.md`` for the module
theory, ``docs/*-analysis.md`` for the measurements.  Those files are written in
Doxygen's Markdown dialect because the Doxygen manual is a first-class
deliverable with a zero-warning acceptance criterion (see ``CLAUDE.md``).

Rather than fork them, this module rewrites that dialect into the one
MkDocs-Material speaks, at build time.  There is no second copy to keep in step,
which is the same argument ``docs/doxygen/md_math_filter.py`` makes for the
LaTeX dialects, one layer down.

What is translated
------------------

=============================== ==============================================
Doxygen                         MkDocs-Material
=============================== ==============================================
``Title\\n=====`` (setext)       ``# Title`` (ATX; ``{#label}`` kept for attr_list)
``[TOC]``                       dropped -- the theme renders its own
``@note`` / ``@warning`` para   ``!!! note`` / ``!!! warning`` admonition
``@subpage x`` / ``@ref x``     a Markdown link, resolved through LinkIndex
``#MFTL_Symbol``                a link into the mkdoxy API reference
``MFTLFunction()``              a link into the mkdoxy API reference
``\\f$ x \\f$`` / ``\\f[ x \\f]``  ``$ x $`` / ``$$ x $$`` for Arithmatex
``@ingroup x``                  dropped -- groups are a Doxygen-only concept here
=============================== ==============================================

Everything else is left exactly as written: fenced code, inline code spans and
math are protected by placeholder before any inline rule runs, so an identifier
inside ``\\texttt{...}`` or inside a shell snippet is never linkified.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
_ATX_LABEL_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*\{#([A-Za-z0-9_]+)\}\s*$")
_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_SETEXT_H1_RE = re.compile(r"^=+\s*$")
_SETEXT_H2_RE = re.compile(r"^-{3,}\s*$")
_LABEL_ONLY_RE = re.compile(r"\{#([A-Za-z0-9_]+)\}")

#: Doxygen paragraph commands mapped onto Material admonition types.
_ADMONITION = {
    "note": ("note", None),
    "warning": ("warning", None),
    "attention": ("warning", "Attention"),
    "remark": ("info", "Remark"),
    "bug": ("bug", None),
    "todo": ("info", "To do"),
    "see": ("info", "See also"),
    "invariant": ("abstract", "Invariant"),
    "par": ("abstract", None),
}
_ADMONITION_RE = re.compile(r"^@(" + "|".join(_ADMONITION) + r")\b[ \t]*(.*)$")


def slugify(text: str) -> str:
    """Reproduce ``pymdownx.slugs.slugify(case='lower')`` for a heading.

    mkdoxy emits its member documentation as ``### function MFTLObjectCreate``,
    so the anchor a cross-reference has to point at is derivable rather than
    discoverable.  Keeping the two in step is why this function exists here
    instead of being guessed at each call site.

    :param text: heading text as it will be rendered.
    :returns: the fragment identifier Material will assign to it.
    """
    text = re.sub(r"[^\w\s-]", "", text.replace("\\", "")).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def relative_url(from_page: str, to_page: str, anchor: str | None = None) -> str:
    """Build a Markdown-source-relative link between two site pages.

    MkDocs resolves links written against the *source* tree, so this returns a
    path ending in ``.md`` rather than a rendered URL.

    :param from_page: site-relative path of the page holding the link.
    :param to_page: site-relative path of the page being linked to.
    :param anchor: optional fragment, without the leading ``#``.
    :returns: the relative link target.
    """
    rel = posixpath.relpath(to_page, posixpath.dirname(from_page))
    if rel == posixpath.basename(to_page) and posixpath.dirname(from_page) == posixpath.dirname(to_page):
        rel = posixpath.basename(to_page)
    return f"{rel}#{anchor}" if anchor else rel


# --------------------------------------------------------------------------
# The link index
# --------------------------------------------------------------------------


@dataclass
class PageAnchor:
    """Where a ``{#label}`` from the Markdown sources ended up in the site."""

    page: str
    anchor: str | None
    title: str


@dataclass
class LinkIndex:
    """Resolves every kind of Doxygen cross-reference to a site-relative link.

    Three namespaces are searched, in this order, because they cannot collide:
    the ``{#label}`` anchors harvested from the Markdown sources, the Doxygen
    *compounds* (files, structs, groups) and the Doxygen *members* (functions,
    macros, typedefs).  The latter two come from the same ``index.xml`` mkdoxy
    parses, so a link can never point at a page mkdoxy did not generate.
    """

    #: ``{#label}`` -> where it lives now.
    anchors: dict[str, PageAnchor] = field(default_factory=dict)
    #: Doxygen compound name (``mftl_object.h``, ``MFTL_Object``, ``mftl_object``)
    #: -> ``(refid, kind)``.
    compounds: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: Doxygen member name -> ``(owning compound refid, kind)``.
    members: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: Directory the mkdoxy pages are generated into.
    api_path: str = "api"

    # -- population ---------------------------------------------------------

    def add_anchor(self, label: str, page: str, anchor: str | None, title: str) -> None:
        """Register a ``{#label}`` found in one of the Markdown sources.

        :param label: the label, without ``{#`` and ``}``.
        :param page: site-relative path of the page it now lives on.
        :param anchor: fragment on that page, or ``None`` for the page itself.
        :param title: heading text, used when a reference supplies no link text.
        """
        self.anchors[label] = PageAnchor(page=page, anchor=anchor, title=title)

    # -- resolution ---------------------------------------------------------

    def _compound_page(self, refid: str) -> str:
        return f"{self.api_path}/{refid}.md"

    def resolve(self, target: str) -> tuple[str, str | None, str] | None:
        """Resolve one reference target.

        :param target: what followed ``@ref`` / ``@subpage`` / ``#`` / the name
            in front of ``()``.
        :returns: ``(page, anchor, default_title)`` or ``None`` when the target
            is unknown, in which case the caller renders it as plain code.
        """
        hit = self.anchors.get(target)
        if hit is not None:
            return hit.page, hit.anchor, hit.title

        compound = self.compounds.get(target)
        if compound is not None:
            return self._compound_page(compound[0]), None, target

        member = self.members.get(target)
        if member is not None:
            refid, kind = member
            anchor = slugify(f"{kind} {target}") if kind in ("function", "define", "typedef", "variable", "enum") else None
            return self._compound_page(refid), anchor, target
        return None


# --------------------------------------------------------------------------
# Phase A -- line-oriented rewrites
# --------------------------------------------------------------------------


def _iter_fence_state(lines: list[str]):
    """Yield ``(line, inside_fence)`` for every line of a Markdown document.

    The fence marker lines themselves are reported as *inside*, so a caller
    that skips fenced content never rewrites the marker either.

    :param lines: the document, split on newlines.
    :returns: generator of ``(str, bool)``.
    """
    fence: str | None = None
    for line in lines:
        match = _FENCE_RE.match(line)
        if fence is None:
            if match:
                fence = match.group(2)[0] * 3
                yield line, True
                continue
            yield line, False
        else:
            if match and match.group(2).startswith(fence):
                fence = None
            yield line, True


def _setext_to_atx(lines: list[str]) -> list[str]:
    """Rewrite setext headings as ATX so attr_list sees a normal heading.

    ``attr_list`` does support setext, but the ``---`` underline is also a
    horizontal rule and a table delimiter, so normalising removes the ambiguity
    rather than relying on Python-Markdown resolving it the same way Doxygen
    did.

    :param lines: document lines.
    :returns: rewritten lines.
    """
    out: list[str] = []
    states = list(_iter_fence_state(lines))
    i = 0
    while i < len(states):
        line, in_fence = states[i]
        nxt, nxt_in_fence = states[i + 1] if i + 1 < len(states) else ("", False)
        if not in_fence and not nxt_in_fence and line.strip() and not line.startswith((" ", "\t", ">", "|")):
            if _SETEXT_H1_RE.match(nxt):
                out.append(f"# {line.strip()}")
                i += 2
                continue
            if _SETEXT_H2_RE.match(nxt) and not line.lstrip().startswith(("-", "*", "+")):
                out.append(f"## {line.strip()}")
                i += 2
                continue
        out.append(line)
        i += 1
    return out


def _convert_math_delimiters(text: str) -> str:
    """Turn Doxygen's ``\\f`` math delimiters into the dollar dialect.

    Arithmatex is configured in generic mode, so ``$ ... $`` and ``$$ ... $$``
    are what reaches MathJax.  The module READMEs already use that dialect --
    they are read on GitHub too -- so only the guide pages are affected.

    :param text: the document.
    :returns: the document with dollar-delimited math.
    """
    text = re.sub(r"\\f\$(.+?)\\f\$", lambda m: f"${m.group(1)}$", text, flags=re.DOTALL)
    text = re.sub(r"\\f\[(.+?)\\f\]", lambda m: f"$$\n{m.group(1).strip()}\n$$", text, flags=re.DOTALL)
    text = re.sub(r"\\f\{([a-z*]+)\}\{(.+?)\\f\}",
                  lambda m: f"$$\n\\begin{{{m.group(1)}}}{m.group(2).strip()}\\end{{{m.group(1)}}}\n$$",
                  text, flags=re.DOTALL)
    return text


def _convert_admonitions(lines: list[str]) -> list[str]:
    """Rewrite ``@note`` / ``@warning`` paragraphs as Material admonitions.

    A Doxygen paragraph command runs to the end of its paragraph, so the block
    is everything up to the next blank line.  Continuation lines are aligned
    under the command in the source; they are dedented and re-indented by four
    so the admonition body is a normal Markdown block.

    :param lines: document lines.
    :returns: rewritten lines.
    """
    out: list[str] = []
    states = list(_iter_fence_state(lines))
    i = 0
    while i < len(states):
        line, in_fence = states[i]
        match = None if in_fence else _ADMONITION_RE.match(line)
        if match is None:
            out.append(line)
            i += 1
            continue

        kind, title = _ADMONITION[match.group(1)]
        body = [match.group(2).strip()] if match.group(2).strip() else []
        i += 1
        continuation: list[str] = []
        while i < len(states) and states[i][0].strip() and not states[i][1]:
            continuation.append(states[i][0])
            i += 1
        if continuation:
            indent = min(len(c) - len(c.lstrip()) for c in continuation)
            body.extend(c[indent:].rstrip() for c in continuation)

        head = f'!!! {kind} "{title}"' if title else f"!!! {kind}"
        out.append(head)
        out.append("")
        out.extend(f"    {b}" if b else "" for b in body)
        out.append("")
    return out


# --------------------------------------------------------------------------
# Phase B -- inline rewrites, with code and math held out
# --------------------------------------------------------------------------

_PROTECT_PATTERNS = [
    re.compile(r"^(?P<body>(?P<f>`{3,}|~{3,}).*?\n(?:.*?\n)*?[ \t]*(?P=f)[ \t]*)$", re.MULTILINE),  # fenced block
    re.compile(r"(?P<body>\$\$.+?\$\$)", re.DOTALL),                                                # display math
    re.compile(r"(?P<body>(?<![\\$])\$[^$\n]+?\$(?!\$))"),                                          # inline math
    re.compile(r"(?P<body>(`+)(?:.|\n)*?\2)"),                                                      # inline code span
]


def _protect(text: str) -> tuple[str, list[str]]:
    """Replace code and math with opaque placeholders.

    :param text: the document.
    :returns: ``(text with placeholders, the extracted fragments)``.
    """
    store: list[str] = []

    def stash(match: re.Match) -> str:
        store.append(match.group("body"))
        return f"\x00{len(store) - 1}\x00"

    for pattern in _PROTECT_PATTERNS:
        text = pattern.sub(stash, text)
    return text, store


def _restore(text: str, store: list[str]) -> str:
    """Put the protected fragments back.

    :param text: the document with placeholders.
    :param store: fragments produced by :func:`_protect`.
    :returns: the document.
    """
    return re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], text)


_REF_RE = re.compile(r"@(ref|subpage)\s+([A-Za-z_][A-Za-z0-9_.:]*)(?:\s+\"([^\"]*)\")?")
_HASH_RE = re.compile(r"(?<![\w`{#])#([A-Za-z_][A-Za-z0-9_]*)")
_CALL_RE = re.compile(r"(?<![\w`])(MFTL[A-Za-z0-9_]*)\(\)")


def _inline_rewrites(text: str, page: str, index: LinkIndex) -> str:
    """Apply the four cross-reference rules to unprotected prose.

    :param text: prose with code and math already replaced by placeholders.
    :param page: site-relative path of the page being generated.
    :param index: the resolver.
    :returns: prose with Markdown links.
    """

    def ref(match: re.Match) -> str:
        target, title = match.group(2), match.group(3)
        hit = index.resolve(target)
        if hit is None:
            return title or f"`{target}`"
        to_page, anchor, default_title = hit
        return f"[{title or default_title}]({relative_url(page, to_page, anchor)})"

    def hashed(match: re.Match) -> str:
        target = match.group(1)
        hit = index.resolve(target)
        if hit is None:
            return f"`{target}`"
        to_page, anchor, _ = hit
        return f"[`{target}`]({relative_url(page, to_page, anchor)})"

    def call(match: re.Match) -> str:
        target = match.group(1)
        hit = index.resolve(target)
        if hit is None:
            return f"`{target}()`"
        to_page, anchor, _ = hit
        return f"[`{target}()`]({relative_url(page, to_page, anchor)})"

    text = _REF_RE.sub(ref, text)
    text = _HASH_RE.sub(hashed, text)
    text = _CALL_RE.sub(call, text)
    return text


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def harvest_anchors(text: str, page: str, index: LinkIndex) -> str | None:
    """Record every ``{#label}`` of one source document in the index.

    Called for all sources *before* any of them is converted, because a
    reference in the first page may point at a label in the last.

    :param text: the raw Doxygen-flavoured document.
    :param page: site-relative path the document will be written to.
    :param index: the index to populate.
    :returns: the document's own title, or ``None`` if it has no heading.
    """
    lines = _setext_to_atx(text.splitlines())
    doc_title: str | None = None
    for line, in_fence in _iter_fence_state(lines):
        if in_fence:
            continue
        match = _ATX_LABEL_RE.match(line)
        if match is None:
            if doc_title is None:
                plain = _ATX_RE.match(line)
                if plain and len(plain.group(1)) == 1:
                    doc_title = plain.group(2)
            continue
        level, title, label = len(match.group(1)), match.group(2), match.group(3)
        is_page = level == 1 and doc_title is None
        if is_page:
            doc_title = title
        index.add_anchor(label, page, None if is_page else label, title)
    return doc_title


def convert(text: str, page: str, index: LinkIndex) -> str:
    """Rewrite one Doxygen-flavoured document into Material Markdown.

    :param text: the raw document.
    :param page: site-relative path it will be written to, used to make links
        relative.
    :param index: a fully populated resolver.
    :returns: the converted document.
    """
    text = _convert_math_delimiters(text)
    lines = _setext_to_atx(text.splitlines())
    lines = [ln for ln in lines if ln.strip() not in ("[TOC]",) and not ln.strip().startswith("@ingroup ")]
    lines = _convert_admonitions(lines)

    body, store = _protect("\n".join(lines))
    body = _inline_rewrites(body, page, index)
    body = _restore(body, store)
    return body.rstrip() + "\n"
