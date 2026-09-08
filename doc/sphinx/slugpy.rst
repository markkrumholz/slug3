.. highlight:: rest

.. _sec-slugpy:

slugpy 
======

slugpy is slug's Python frontend: it contains utilities to run simulations,
and to read and manipulate SLUG outputs. This page details how to use slugpy
to carry out a range of common tasks, while :ref:`sec-slugpy-full` documents
the full API.

Controlling Simulation Physics
------------------------------

As discussed in :ref:`ssec-running-python`, one can run a SLUG simulation
from Python by doing::

    import slugpy
    sim_controls = slugpy.SimControls(...)
    sim_result = slugpy.run_sim(sim_controls)

The ``slugpy.SimControls`` object can be constructed by reading an input
deck, either on-disk or directly from a toml table (see :ref:`sec-slugpy-full`),
but it is also possible to control its properties directly from Python. For
example, to change the IMF used in a simulation one can do::

   sim_controls.setIMF("path/to/pdf.toml")

where "pdf.toml" is a :ref:`ssec-pdf-files`. To add a photometric filters
to a simulation, one can could do::

   sim_controls.filters.addFilter("filter_name")

where "filter_name" is the name of a filter. Every parameter in
:ref:`sec-parameters` has a corresponding representation in
``slugpy.SimControls``, and methods to modify or query these parameters.
In most cases the current values of parameters are stored as properties
of the class (e.g., sim_controls.imf), and these can either be changed
directly by assigning them to an object of the same type, or, more 
conveniently, by calling a setter method (e.g., sim_controls.setIMF)
that accepts a string input. See :ref:`sec-slugpy-full` for full
documentation of the ``SimControls`` class and its various options for
controlling simulations. Interactive control of parameters in Python makes
it possible, for example, to automate the process of running simulations
with different physics, or to interactively explore how results change
as the physics changes. 


Reading Outputs
---------------

As described in :ref:`ssec-outputs-reading`, the function ``read`` in slugpy
provides a lazy-reader for SLUG HDF5 outputs. The 


Interrogating Tracks and Isochrones
-----------------------------------


Converting Between Photometric Systems
--------------------------------------