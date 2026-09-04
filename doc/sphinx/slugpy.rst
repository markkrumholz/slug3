.. highlight:: rest

.. _sec-slugpy:

slugpy -- The Python Helper Library
=====================================

slugpy is slug's Python frontend: it wraps the compiled C++ core
(exposed as the private ``_slug`` extension module, built by CMake --
see :doc:`cpp_api/cpp_api_root` for its own API documentation) and
adds pure-Python analysis code of its own for reading and post-
processing slug output files.

Full Documentation of slugpy
-------------------------------

The pure-Python layer -- reading/post-processing slug output files,
plus a handful of names re-exported directly from ``_slug`` (classes
are documented separately below, under their own ``slugpy._slug.*``
module heading, since Sphinx's autodoc looks for members under the
module that actually defines them, not one that merely re-exports
them):

.. automodule:: slugpy
   :members:
   :imported-members:

The compiled core (``slugpy._slug``)
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

.. automodule:: slugpy._slug
   :members:
   :undoc-members:
