"""
Sphinx configuration for slug's documentation.

Documents both halves of the codebase from the same build: the Python
frontend (slugpy) via sphinx.ext.autodoc/napoleon, and the C++ core
(src/) via Exhale, which drives Doxygen+Breathe to produce a full,
natively-organized API site (separate Namespace/Class/File listings,
each alphabetical, rather than one flat page) instead of parsing C++
itself. Building this requires slugPython to have already been built
(`cmake --build build --target slugPython`) -- slugpy/__init__.py
imports the compiled _slug extension at module load time, so autodoc
cannot document slugpy without it actually being importable.

Must be built from this directory (doc/sphinx/), e.g. via `make html`
(the final HTML lands in doc/html, not doc/sphinx/_build/html -- see
this directory's own Makefile) or `sphinx-build -b html . ../html` run
from here -- Exhale's own Doxygen invocation inherits this process's
cwd, and the relative paths below (INPUT, doxygenStripFromPath) are
written assuming it.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import os
import sys
import textwrap

from docutils import nodes

# slugpy itself, for autodoc -- doc/sphinx/../.. is the repo root.
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

# -- Project information -------------------------------------------------
project = "slug"
copyright = "2026, Mark Krumholz, Michele Fumagalli, et al."
author = "Mark Krumholz, Michele Fumagalli, et al."
version = "3.0"
release = "3.0"

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    # slugpy's docstrings are numpy-style (AGENTS.md's own Coding
    # style section) -- napoleon translates those into what autodoc
    # expects, the same role it plays for any numpydoc-convention
    # project.
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    # breathe renders individual C++ entities from Doxygen's XML;
    # exhale (below) is what actually organizes them into the full,
    # natively-structured site (Namespace List, Class List, File List,
    # etc.) -- breathe alone only gives you directives to hand-place
    # one entity (or, via doxygenindex, literally everything at once,
    # unsorted and undivided) into an .rst page yourself.
    "breathe",
    "exhale",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]
master_doc = "index"
pygments_style = "sphinx"

# The pybind docstrings (src/pybind/Bind*.cpp) use "Throws" and
# "Details" section headers rather than numpydoc's own "Raises"/
# "Notes" (see AGENTS.md's own Coding style section for the
# convention) -- without these, Napoleon leaves those sections as
# unconverted plain text, which can butt up against a properly-
# converted field list right above it with no blank line in between
# and make docutils reject the page outright.
napoleon_custom_sections = [("Throws", "Raises"), ("Details", "notes")]

# -- Options for HTML output -----------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Breathe / Exhale (C++ API docs) ---------------------------------------
# breathe_projects/breathe_default_project are required by Breathe
# itself; Exhale also reads breathe_projects[breathe_default_project]
# to know where to point Doxygen's own OUTPUT_DIRECTORY (exhale_args
# below is deliberately not allowed to set OUTPUT_DIRECTORY itself,
# precisely so these two can never disagree).
breathe_projects = {"slug": os.path.abspath(os.path.join("doxygen", "xml"))}
breathe_default_project = "slug"

exhale_args = {
    # Generated .rst files land here, under the Sphinx source tree;
    # index.rst's own toctree links directly to this tree's root page
    # (cpp_api/cpp_api_root) -- there is deliberately no hand-written
    # cpp_api.rst wrapper page above it, since that would just repeat
    # rootFileTitle as a redundant extra level in the sidebar.
    "containmentFolder": "./cpp_api",
    "rootFileName": "cpp_api_root.rst",
    "rootFileTitle": "The slug C++ API",
    "afterTitleDescription": textwrap.dedent("""
        This section documents slug's C++ core (the ``src/`` tree,
        excluding the vendored third-party libraries under
        ``src/extern`` and the compiled-extension glue under
        ``src/pybind`` -- see :doc:`/slugpy` for that), generated
        directly from its own Doxygen docstrings via `Exhale
        <https://exhale.readthedocs.io/>`_, which organizes them the
        same way Doxygen's own native HTML output does -- separate,
        alphabetized Namespace, Class, and File listings -- rather
        than one undivided page.
    """),
    # Strips the path prefix up through the repo root from every
    # displayed file path in the File List (e.g. "src/tracks/
    # Tracks3D.hpp" instead of an ugly full absolute path) -- resolved
    # relative to this file's own directory (doc/sphinx/).
    "doxygenStripFromPath": "../..",
    # A collapsible class/file/namespace hierarchy in the sidebar,
    # matching native Doxygen's own tree-view feel.
    "createTreeView": True,
    # Let Exhale drive Doxygen itself (rather than the other way
    # around, as a separately-run doc/Doxyfile would), so it can
    # guarantee its own required settings (OUTPUT_DIRECTORY,
    # STRIP_FROM_PATH, matching breathe_projects/doxygenStripFromPath
    # above) never drift out of sync with what it actually generates
    # pages for -- see exhale's own exhaleUseDoxyfile docs for why
    # exhaleDoxygenStdin, not a separate physical Doxyfile, is the
    # encouraged way to do this.
    "exhaleExecutesDoxygen": True,
    # INPUT: every src/ subdirectory except extern (vendored third-
    # party libraries -- pybind11, pcg-cpp, tomlplusplus, mdspan --
    # not slug's own code to document) and pybind (the compiled-
    # extension glue: its own file-scope *Docstring string_view
    # constants are giant multi-line raw strings that Sphinx's C++
    # domain parser can't parse as declaration initializers, and that
    # text is already surfaced properly through slugpy's own
    # automodule-generated Python docs -- see slugpy.rst -- which is
    # the actual user-facing home for it, so nothing is lost by
    # excluding it here). EXTRACT_ALL = NO matches AGENTS.md's own
    # Coding style convention (every public-facing class/method/
    # function gets a Doxygen docstring): document only what is
    # actually documented, rather than synthesizing entries for every
    # declaration Doxygen can see.
    "exhaleDoxygenStdin": textwrap.dedent("""
        INPUT              = ../../src
        FILE_PATTERNS      = *.hpp
        EXCLUDE            = ../../src/extern ../../src/pybind
        EXTRACT_ALL        = NO
    """),
}

# Breathe's own default role for a bare `:cpp:...:` cross-reference
# with no explicit domain.
primary_domain = "cpp"
highlight_language = "cpp"


def _fix_breathe_dash_placeholder(app, doctree, docname):
    """
    Replace Breathe's literal "&#8212;" text with a real em dash.

    Doxygen converts "--"/"---" in a doc comment into its own <ndash/>
    /<mdash/> XML nodes; Breathe renders *both* as the literal text
    "&#8212;" (see its parser/compound.py) instead of an actual dash
    character, so docutils later escapes the "&" when writing HTML,
    leaving the raw string "&#8212;" visible on the page. This is a
    known upstream Breathe bug (unrelated to how slug's own C++ doc
    comments are written -- see AGENTS.md's own Coding style section
    for that "--" convention), fixed here by swapping the placeholder
    for a real U+2014 character before any writer serializes it.
    """
    for text_node in list(doctree.findall(nodes.Text)):
        if "&#8212;" in text_node.astext():
            fixed = text_node.astext().replace("&#8212;", "—")
            text_node.parent.replace(text_node, nodes.Text(fixed))


def setup(app):
    app.connect("doctree-resolved", _fix_breathe_dash_placeholder)
