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

As described in :ref:`ssec-output-reading`, the function ``read`` in slugpy
provides a lazy-reader for SLUG HDF5 outputs, which can be accessed by doing

   .. code-block:: python
      import slugpy
      sim_result = slugpy.read("path/model_name")

The lazy-reader is an object of class ``slug_reader``, and once it is created,
users can access the data via its properties. Some of the key properties (not
an exhaustive list -- :ref:`sec-slugpy-full`) include:

* ``input_deck``: a copy of the input deck used to produce this output.
* ``clusters``: a dict-like interface to the clusters group's datasets, e.g., ``sim_result.clusters['target_mass']``.
* ``cluster_spectra``: a dict-like interface to the cluster_spectra group's datasets, e.g., ``sim_result.cluster_spectra['spec']`` or ``sim_result.cluster_spectra['neb_lines']``. Append ``_neb`` for nebular output, e.g., ``sim_result.cluster_spectra['spec_neb']``, ``_ex`` for extincted output, and ``_neb_ex`` for output with both nebular processing and extinction.
* ``cluster_phot``: a dict-like interface to the cluster_phot group's data, accessible per filter, e.g., ``sim_result.cluster_phot['HST.ACS_WFC.F435W']``; as with spectra, append ``_neb``, ``_ex``, or ``_neb_ex`` to the filter name to access the outputs with nebular processing, extinction, or both.
* ``cluster_cloudy``: a dict-like interface to the cluster_cloudy group's data, e.g., ``sim_result.cluster_cloudy['spec_trans_emit']``. See :ref:`sec-cloudy-slug`.
* ``galaxy``, ``galaxy_spectra``, ``galaxy_phot``, ``galaxy_cloudy``: same as the ``cluster*`` fields of the same name, but for galaxy outputs.
* ``filters``: list of all photometric filters available in this data set.
* ``controls``: a ``SimControls`` object build from this data set's input deck.

Not all of these data will be present in every file; for example, if no photometry
was requested for the run, then ``cluster_phot`` and ``galaxy_phot`` will be ``None``.
All data will be returned as ``astropy.Quantity`` objects with correct units attached.

In addition to providing access to data, the lazy-reader provides convenience functions
for manipulating it:

* ``slug_reader.get_cluster(uid)``: this method accepts as input the unique ID number of any cluster in the data set and returns a live-constructed ``Cluster`` object. This object can be examined to access data not already in the outputs, for example generating spectra or photometry at times not in the original output. See :ref:`sec-slugpy-full` for a full listing of all of ``Cluster``'s methods.
* ``slug_reader.get_filter(filter_name)``: this method accepts as input the name of any filter present in the data set and returns a live-constructed ``Filter`` objects. This object contains all information about the filter, such as its full response curve. See :ref:`sec-slugpy-full` for details on the ``Filter`` class.
* ``slug_reader.phot_convert(phot_system)``: this method converts any loaded photometric data from the current photometric system to the system named by ``phot_system``, which can be "Flambda", "Fnu", "ST", "AB", or "Vega"; see :ref:`sec-phot` for an explanation of the photometric systems.

Interrogating Tracks and Isochrones
-----------------------------------

The slugpy module provides two convenience methods for examining stellar tracks and
isochrones.

The ``compute_isochrones`` function takes as input a list of times and several other
optional arguments (see :ref:`sec-slugpy-full` for the full API) and returns an array of
stellar masses and a dict of arrays containing the properties of those stars at the
requested times; the properties available are the present-day mass (accounting for
mass loss), mass loss rate, effective temperature, luminosity, surface abundances of
H, He, C, N, and O, and the photometric output in any number of specified filters.

The ``compute_tracks`` function operates similarly, except that it takes as input
a list of stellar masses (and other optional arguments) and returns an array of
times and a dict containing the properties of a star born with each stellar mass
at each time. The properties available are the same as for ``compute_isochrones``.