.. highlight:: rest

.. _sec-yields:

Yields
======

SLUG's Model for Yields
-----------------------

SLUG has the ability to compute the nucleosynthetic yields produced by the stars it
simulates. Yields are described in terms of the *channel* that produces them, where
a channel means a particular stellar process or evolutionary phase. The channels
currently included in SLUG are:

    * ``massive_star_winds``: the winds produced by massive stars prior to supernova
      or collapse to black holes
    * ``ccsn``: core-collapse supernovae

Channels are described in terms of the isotopes they produce when a star of a given
mass dies -- it is assumed that all mass return occurs instantaneously upon stellar
death, rather than being resolved in time, for the practical reason that available
tabulations of stellar nucleosynthetic yields almost never include time-resolved
predictions for element return rates, and even if they did these predictions would
not necessarily be compatible with the evolutionary tracks used to compute stellar
spectral and photometric output. This approximation means that, formally, the
yields for a particular channel are characterized by the yield function
:math:`y_X(m, Z_\mathrm{Fe})` that describes the mass of isotope
:math:`X` returned to the ISM by a star of initial mass :math:`m` and iron
metallicity :math:`Z_\mathrm{Fe}` upon its death.

Note that the yield as we define it here is the total mass returned, not the net
increase in mass after subtracting off the mass that went into forming the star, and
thus :math:`y_X(m, Z_\mathrm{Fe})` is strictly positive. Thus for example
yields of :math:`^1\mathrm{H}` are positive, even though nuclear reactions in stars
almost always mean that on net they reduce the amount of :math:`^1\mathrm{H}` in the
Universe by converting it to heavier elements.

For a given channel, the function :math:`y_X(m, Z_\mathrm{Fe})` is generally
zero outside some mass range :math:`(m_\mathrm{min}, m_\mathrm{max})`, outside of
which the process described by that particular channel ceases to occur -- for example
there is a minimum stellar mass for core-collapse supernovae, and :math:`y_X(m)` is
zero for masses below this range. The values of :math:`m_\mathrm{min}` and
:math:`m_\mathrm{max}` can be chosen by the user, or left to the defaults provided by
the individual models used to compute yields -- see :ref:`ssec-yield-models`.

Computing Yields
----------------

When run with nucleosynthetic yields enabled, SLUG will compute :math:`y_X(m)` for
every isotope and every yield channel requested (see :ref:`ssec-parameters-yields` in
:ref:`sec-parameters`) and return the result in the output file -- see :ref:`sec-output`.
For stars being treated individually and stochastically (see
:ref:`ssec-pdfs-and-monte-carlo`), the yield is computed star-by-star and summed to
produce the final yield :math:`Y_X` at any chosen output time :math:`t`.

For stars that are not being treated stochastically, SLUG computes the yield by evaluating
integrals over the continuous stellar populations. For cluster-type simulations where the
stellar population is all the same age (see :ref:`ssec-cluster-vs-galaxy`), SLUG computes the
total yield returned by a population of age :math:`t` by evaluating

.. math:: Y_X(t) = \frac{M_*}{\left\langle m\right\rangle} \int_{-\infty}^{\infty} \left[\int_0^\infty y_X(m, Z_\mathrm{Fe}) \Theta(t - t_\mathrm{life}(m, Z_\mathrm{Fe})) \frac{dn}{dm} \, dm\right] \frac{dp}{dZ_\mathrm{Fe}} \, dZ_\mathrm{Fe},

where :math:`M_*` is the total mass of the stellar population, :math:`\langle m\rangle` is the
mean stellar mass computed from the IMF :math:`dn/dm`, :math:`\Theta(x)` is the Heaviside step
function (equal to unity for :math:`x > 0`, and zero for :math:`x < 0`),
:math:`t_\mathrm{life}(m, Z_\mathrm{Fe})` is the stellar lifetime as a function of
stellar initial mass and Fe metallicity, and :math:`dp/dZ_\mathrm{Fe}` is the
distribution of Fe metallicity for the stellar population. If only some of the IMF is being treated
non-stochastically, the range of integration is limited to the range being treated
non-stochastically, and the results of this integral are added to the results computed
star-by-star over the stochastic mass range. This integral is modified in the presence
of :ref:`ssec-radioactive-decay`.

For composite stellar populations described by a star formation rate :math:`\dot{M}_*(t)` as
a function of time :math:`t`, SLUG instead evaluates the total yield returned up to time
:math:`t` as

.. math:: Y_X(t) = \int_0^t \dot{Y}_X(t') \, dt'

where :math:`\dot{Y}_X(t)` is the instantaneous rate of mass return at time :math:`t`, given
by

.. math:: \dot{Y}_X(t) = \frac{1}{\left\langle m \right\rangle} \int_{-\infty}^{\infty} \left[\int_0^t \dot{M}_*(t - t') \sum_i y_X(m_{i}(t'), Z_\mathrm{Fe}) \left|\frac{dt_\mathrm{life}}{dm}\right|_{m_{i}(t')}^{-1} \left.\frac{dn}{dm}\right|_{m_{i}(t')} \, dt'\right] \frac{dp}{dZ_\mathrm{Fe}} \, dZ_\mathrm{Fe}.

Here :math:`m_i(t')` is one of the (possibly multiple) solutions to the implicit equation

.. math:: t_\mathrm{life}(m, Z_\mathrm{Fe}) = t',

i.e., :math:`m_i(t')` is the mass of the star whose lifetime is :math:`t'`, and if there are
no such stars the integrand is taken to be zero.

.. _ssec-yield-models:

Standard Yield Models
---------------------

The list of yield models available for use in SLUG is given in a registry file.
The default registry, which is included in the repository, is ``data/yields/yields.toml``.
The registry is a human-readable description of the available yields. Yields are
organized by channel, with the list of available channels provided in the ``channels``
keyword at the top level of the registry, and then the models available for each channel
listed in the ``models`` keyword in the table for that channel.

The SLUG standard data files provide the following yield models. For each, the
default mass range is the range of initial stellar masses covered by the tabulated
yields (the yield is zero outside this range unless the ``m_min`` and ``m_max``
keywords are used to change it -- see :ref:`ssec-parameters-yields` in
:ref:`sec-parameters`), and the [Fe/H] range is the range that the [Fe/H] of the
simulated stellar population, set by the ``FeH`` keyword in the ``[stars]`` section,
must lie within. Yields at intermediate masses and [Fe/H] values are obtained by
linear interpolation. Unlike the stellar tracks, yield models do not cover a
range of rotation rates or [alpha/Fe] values, so the ``v_vcrit`` and ``alphaFe``
parameters have no effect on which yields are used.

* For the ``ccsn`` channel:

  * ``sukhbold16``: the core-collapse supernova yields of `Sukhbold et al. (2016) <https://ui.adsabs.harvard.edu/abs/2016ApJ...821...38S/abstract>`_. The default mass range is :math:`9 - 120\,\mathrm{M}_\odot`, sampled at 200 masses, and the yields are available at solar metallicity only, [Fe/H] = 0, so this model can be used only for simulations with exactly that [Fe/H]. Progenitors that fail to explode return no mass through this channel, and their supernova yield is exactly zero; this is true of roughly half of the tabulated masses, lying between :math:`15` and :math:`100\,\mathrm{M}_\odot`. The mass that such stars lose in winds before collapsing is provided by the ``massive_star_winds`` channel.
  * ``kobayashi06_11``: the core-collapse supernova yields of `Kobayashi et al. (2006) <https://ui.adsabs.harvard.edu/abs/2006ApJ...653.1145K/abstract>`_ and `Kobayashi et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011MNRAS.414.3231K/abstract>`_. The default mass range is :math:`13 - 40\,\mathrm{M}_\odot`, sampled at 13, 15, 18, 20, 25, 30, and 40 :math:`\mathrm{M}_\odot`, and the yields are available at [Fe/H] = -1.30, -0.70, and 0.0 (Z = 0.001, 0.004, and 0.02, with a Solar value of Z = 0.02). This model provides supernova yields only, and is not available for the ``massive_star_winds`` channel.
  * ``limongi_chieffi18_v0``, ``limongi_chieffi18_v150``, ``limongi_chieffi18_v300``: the core-collapse supernova yields of `Limongi & Chieffi (2018) <https://ui.adsabs.harvard.edu/abs/2018ApJS..237...13L/abstract>`_ (their recommended set) for stars with initial rotation velocities of 0, 150, and 300 km/s, respectively. Because these velocities are absolute values rather than fractions of the critical speed, they do not correspond to the :math:`v/v_\mathrm{crit}` values of the stellar tracks, and are simply treated as three independent yield models. The default mass range for all three is :math:`13 - 120\,\mathrm{M}_\odot`, sampled at 13, 15, 20, 25, 30, 40, 60, 80, and 120 :math:`\mathrm{M}_\odot`, and the yields are available at [Fe/H] = -3, -2, -1, and 0. In these models stars above :math:`25\,\mathrm{M}_\odot` collapse directly to black holes without exploding, so the supernova yield is exactly zero at masses of 30 :math:`\mathrm{M}_\odot` and above, and non-zero only for the 13, 15, 20, and 25 :math:`\mathrm{M}_\odot` grid points.

* For the ``massive_star_winds`` channel:

  * ``sukhbold16``: the wind yields of the same progenitor models as the ``sukhbold16`` supernova yields above, i.e., the mass returned by stellar winds before the star explodes or collapses. The default mass range is :math:`9 - 120\,\mathrm{M}_\odot`, sampled at 200 masses, and the yields are available at [Fe/H] = 0 only. Unlike the supernova yields, these are non-zero at every tabulated mass, including for progenitors that fail to explode.
  * ``limongi_chieffi18_v0``, ``limongi_chieffi18_v150``, ``limongi_chieffi18_v300``: the wind yields of the same `Limongi & Chieffi (2018) <https://ui.adsabs.harvard.edu/abs/2018ApJS..237...13L/abstract>`_ models as the supernova yields above, for initial rotation velocities of 0, 150, and 300 km/s. The default mass range is :math:`13 - 120\,\mathrm{M}_\odot` and the yields are available at [Fe/H] = -3, -2, -1, and 0, exactly as for the supernova yields. Since stars above :math:`25\,\mathrm{M}_\odot` do not explode in these models, the wind yield at those masses is the star's entire yield.

The full list of references for each yield model, which should be cited in any
publication using that yield model, is provided in the ``data/yields/yields.toml``
registry.


.. _ssec-adding-yield-models:

Adding New Yield Models
-----------------------

Users are free to add additional yield models to SLUG, and do so without modifying the
source code. All that is required is to create an HDF5 file containing the yield
information that follows the layout specification described below, and then to
register that model in a `TOML <https://toml.io/en/>`_ registry, either by
adding a new entry to the standard registry or by making a new registry file -- the
choice of registry file can be controlled by the ``registry`` keyword in the ``[yields]``
section of the input deck (see :ref:`ssec-parameters-yields` in :ref:`sec-parameters`).

The required format for the HDF5 yield file is as follows.

A yield file contains one top-level HDF5 group for each channel that the model
provides yields for. The name of each group must be exactly the name of the channel,
``ccsn`` or ``massive_star_winds``. A file may contain one of these groups or both;
for example ``kobayashi06_11.h5`` contains only ``ccsn``, whereas ``sukhbold16.h5``
contains both. Each channel group must have:

* A dataset ``masses`` (a 1D array of length :math:`n_\mathrm{mass}`): the initial
  mass, in :math:`\mathrm{M}_\odot`, of every star for which yields are tabulated.
  The masses must be listed in strictly increasing order -- SLUG checks this and
  raises an error if the list is empty or is not strictly ascending -- and they need not
  be evenly spaced. This is the same for every [Fe/H] the channel provides, so a
  channel that provides yields at more than one [Fe/H] must use the same set of masses
  for all of them. The first and last entries of this array set the default mass range
  of the model.
* Datasets ``isotope_z`` and ``isotope_a`` (1D arrays of the same length,
  :math:`n_\mathrm{iso}`): the atomic number :math:`Z` and mass number :math:`A` of every
  isotope whose yield is tabulated. Every :math:`(Z, A)` pair must be present in SLUG's
  own table of isotopes, ``data/elem/isotopes.h5``, which also supplies the radioactive
  decay data used to treat unstable isotopes (see :ref:`ssec-radioactive-decay`); SLUG
  raises an error if it finds an isotope that is not in that table. Different models are
  not required to tabulate the same isotopes: when several models are used together, SLUG
  works with the union of their isotopes, and any isotope that a particular model does not
  tabulate is treated as having zero yield from that model.
* One HDF5 group for each [Fe/H] value at which yields are tabulated. The name given to
  each of these groups is arbitrary and is never interpreted by SLUG -- the standard files
  name them ``feh_<value>``, e.g. ``feh_-1`` -- because SLUG identifies them instead by
  looking for direct children of the channel group that have an attribute ``Fe_H``, as
  described below, and ignores any child that does not. Each of these groups must have:

  * An attribute ``Fe_H`` (a single float): the [Fe/H] value that this group's
    yields correspond to. This attribute is required, and no two groups within a channel
    may have the same value.
  * A dataset ``yield`` (a 2D array of shape :math:`(n_\mathrm{iso}, n_\mathrm{mass})`,
    i.e., with one *row* per isotope and one *column* per mass -- SLUG checks this shape and
    raises an error if it does not match ``isotope_z``, ``isotope_a`` and ``masses``):
    the mass, in :math:`\mathrm{M}_\odot`, of each isotope returned to the ISM by a star
    of each initial mass, in the same isotope order as ``isotope_z`` and
    ``isotope_a`` and the same mass order as ``masses``. As described in
    :ref:`sec-yields`, this is the total mass returned, not the net change in the mass of
    that isotope, and is therefore non-negative. A yield of exactly zero is
    meaningful, for example for a star that collapses to a black hole without exploding and
    so returns nothing through the ``ccsn`` channel.

A file may also have the top-level attributes ``reference`` and ``reference_url`` (strings)
recording the publication the yields come from; these are written by the tools described
below, but are purely descriptive and are never read by SLUG itself.

SLUG interpolates linearly in both mass and [Fe/H] between the tabulated values. The range
of [Fe/H] provided by the file must cover the whole range of [Fe/H] of the simulated stellar
population; if it does not, SLUG raises an error when it reads the file. In particular, a
model that provides yields at only a single [Fe/H] can be used only for a simulation in which
[Fe/H] has exactly that value.

The required format for a registry entry is as follows.

A yield registry is a TOML file with one top-level table for each channel, named with the
name of that channel (e.g. ``[ccsn]``), containing a key ``models`` that gives an array of
strings, one per model that the registry provides for that channel; the default registry,
``data/yields/yields.toml``, lists the standard models described above. For every name
appearing in ``models`` in a channel's table, the registry must also contain a table named
with the channel followed by the model, separated by a dot (e.g. ``[ccsn.sukhbold16]``), with
at least the following key:

* ``file`` (a string): The path to this model's HDF5 file, resolved relative to the
  directory containing the registry file itself (not the current working directory), so that
  a registry and the yield files it describes can be moved together as a unit. A model that
  provides more than one channel gives the same file name in each of its channel entries,
  e.g. in both ``[ccsn.sukhbold16]`` and ``[massive_star_winds.sukhbold16]``.

A registry entry may also include any of the following optional keys, which SLUG itself never
reads but which are included in the standard registry for documentation and citation purposes,
and are recommended for any new model added in the same way:

* ``reference``: A string giving the citation for the publication that should be cited by
  anyone using this model, or an array of strings if there is more than one.
* ``reference_url``: A string giving a URL for ``reference``, or an array of strings, one per
  entry of ``reference``, if there is more than one.
* ``Fe_H``: An array of numbers, the [Fe/H] values this model provides. SLUG does not use
  this key: the [Fe/H] values it actually uses are always read directly from the ``Fe_H``
  attribute of each group in the HDF5 file. It exists purely to let a user discover a
  model's [Fe/H] coverage without opening the HDF5 file.
* ``masses``: An array of numbers, the initial stellar masses in :math:`\mathrm{M}_\odot` at
  which this model provides yields, likewise provided purely for the user's benefit.

Finally, the registry may include a top-level ``name`` key (a string), a human-readable
title for the registry as a whole, and a top-level ``channels`` key (an array of strings)
listing the channels the registry describes; like the per-entry optional keys above, these
are purely descriptive and are never read by SLUG itself.

Examples of how to add yields are provided by the scripts in the ``data/tools/yields``
directory; the scripts there were the ones used to populate the standard yield models.
The script ``import_yield_tables.py`` converts a directory of plain-text yield tables,
one for each initial mass, into an HDF5 file in the format above and adds the
corresponding entries to the registry. Each table must be named ``s<mass>.yield_table``,
where ``<mass>`` is the initial mass in :math:`\mathrm{M}_\odot` (e.g. ``s18.2.yield_table``),
and consists of a header row followed by one row for each isotope, each of which gives the
isotope's name, formed from its element symbol and its mass number with no space between
them (e.g. ``fe56``), followed by its yield in each column. The header row may be any of
``[isotope] [ejecta] [wind]``, ``[isotope] [wind]``, or ``[isotope] [ejecta]``, which
determine whether the columns supply the ``ccsn`` yields, the ``massive_star_winds`` yields,
or both. Any combination of isotope and mass that a table has no row for is written as a yield of
exactly zero, as is the supernova yield for a mass whose table has no ``[ejecta]`` column
at all, which is taken to be a star that did not explode. The script is run once for each
[Fe/H] the model provides, adding that [Fe/H] to the model's file each time; the model's name,
its reference and reference URL, and the [Fe/H] value are given as command-line arguments
(``--name``, ``--reference``, ``--reference-url``, and ``--feh``). Run the script with the
``--help`` flag for a full description. For data that are not in this format, the script
``fetch_limongi_chieffi18.py``, which downloads the tables of `Limongi & Chieffi (2018)
<https://ui.adsabs.harvard.edu/abs/2018ApJS..237...13L/abstract>`_ from the CDS archive,
reformats them, and then uses ``import_yield_tables.py`` to write them, provides an example
of how to build a model from another source.

.. _ssec-radioactive-decay:

Radioactive Decay
-----------------

Some of the isotopes whose yields can be computed are unstable, for example :math:`^{26}\mathrm{Al}`
or :math:`^{60}\mathrm{Fe}`. SLUG offers a choice as to how to treat these. One option is
simply to report the total mass of each isotope produced, ignoring the fact that some of those
isotopes will decay over time. The other option is to self-consistently compute the effects of
radioactive decay by solving the generalized Bateman equations

.. math:: \frac{dN_i}{dt} = -\lambda_i N_i(t) + \sum_{j\neq i} b_{j\to i} \lambda_j N_j(t),

where :math:`N_i` is the number of nuclei of isotope :math:`i`, :math:`\lambda_i` is the
radioactive decay rate of that isotope, and :math:`b_{j\to i}` is the branching ratio for
isotope :math:`j` to isotope :math:`i`, to update the masses of each isotope as a function
of time. See :ref:`ssec-parameters-yields` in :ref:`sec-parameters` for the keywords used to
control which treatment is adopted. SLUG implements its numerical solution to the Bateman
equations using a matrix exponentiation method.