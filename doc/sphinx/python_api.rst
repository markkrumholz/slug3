.. _sec-slugpy-full:

The slug Python API
===================

slugpy wraps the compiled C++ core (exposed as the private ``_slug``
extension module, built by CMake -- see :doc:`cpp_api/cpp_api_root`
for its own API documentation) and adds pure-Python analysis code of
its own for reading and post-processing slug output files.

The Pure Python Layer
---------------------

The pure-Python layer -- reading/post-processing slug output files,
plus a handful of top-level convenience functions
(``compute_isochrones``, ``compute_tracks``, ``phot_convert``,
``read``, ``run_sim``). ``slugpy/__init__.py`` also wildcard-re-exports
every class and function from ``_slug`` (so e.g. ``slugpy.Cluster``
works directly, not just ``slugpy._slug.Cluster``), but those are
documented separately below, under their own ``slugpy._slug`` module
heading, since Sphinx's autodoc looks for members under the module
that actually defines them, not one that merely re-exports them.
``:members:`` is given this section's own explicit name list, rather
than a bare flag plus ``:imported-members:``, specifically to keep
that ``_slug`` wildcard re-export out of this listing --
``:imported-members:`` can't tell an explicit, named import (the five
functions above) apart from a wildcard one, so it would otherwise
pull every ``_slug`` class in here too, duplicating the listing below:

.. automodule:: slugpy
   :members: compute_isochrones, compute_tracks, phot_convert, read, run_sim

The Compiled Core (``slugpy._slug``)
---------------------------------------

Every class and function listed here is also importable directly from
``slugpy`` itself (e.g. ``slugpy.SimControls``, not just
``slugpy._slug.SimControls``) -- see this module's own docstring for
why it's implemented as a private, wildcard-re-exported submodule
rather than living in ``slugpy`` directly.

``:undoc-members:`` is needed here (unlike the automodule above): a
``py::class_<...>(m, "Name")`` binding with no class-level docstring
argument (only per-method docstrings via ``.def(...)``, the
convention throughout ``src/pybind``) leaves the class's own
``__doc__`` as ``None``, and autodoc skips undocumented members by
default -- without this, most of the classes below (anything that
happens not to pass a class-level docstring, like ``SimControls`` or
``Cluster``) would silently vanish from this page even though their
own methods are all individually documented.

``:special-members: __init__`` is needed for the same underlying
reason, applied to constructors specifically: autodoc treats
``__init__`` (like every dunder method) as a "special member", which
it excludes by default even with ``:members:``/``:undoc-members:`` on
its own. Every constructor in ``src/pybind`` is bound via
``py::init<...>(docstring, ...)`` with its own numpydoc-style
docstring (the same convention as every other method), so this is
purely about telling autodoc to surface documentation that already
exists, not about generating anything new.

.. automodule:: slugpy._slug
   :members:
   :undoc-members:
   :special-members: __init__
