.. highlight:: rest

.. _sec-cloudy-slug:

Cloudy Nebular Processing
=========================

SLUG stochastically generates stellar spectra, and it includes an
approximate computation of the nebular lines produced when those
photons interact with the interstellar medium. However, this
approximation ignores a number of potentially important effects, and
does not properly account for the stochastic nature of the stellar
spectra. To perform a much more accurate calculation, SLUG includes an
automated interface to `cloudy <http://nublado.org/>`_ (`Gunasekera et
al., 2025, RMxAA, 61, 120
<https://ui.adsabs.harvard.edu/abs/2025RMxAA..61c.120G/abstract>`_). This can be
used to post-process the output of a SLUG run in order to compute
nebular emission.

Using Cloudy on SLUG Outputs
----------------------------

The basic steps involved in running cloudy on SLUG outputs are as follows:

#. First download and install cloudy, following the directions on the `cloudy website <http://nublado.org/>`_.
#. Set the environment variable ``CLOUDY_DIR`` to the directory containing the compiled ``cloudy.exe`` executable (optional, but recommended).
#. Edit the :ref:`ssec-cloudy-slug-template` file if you want to use a non-default setup for Cloudy (optional).
#. If the SLUG outputs do not already have a ``slug_reader`` lazy-reader attached to them, read the SLUG simulation on which you want to run cloudy (see :ref:`sec-slugpy`):

   .. code-block:: python

      import slugpy
      sim_result = slugpy.read("path/model_name")

#. Use the ``slug_reader.run_cloudy`` method to run cloudy on either the cluster or galaxy spectra stored in the SLUG outputs:

   .. code-block:: python

      sim_result.run_cloudy("cluster")     # or sim_result.run_cloudy("galaxy")

#. The cloudy output will be added to the HDF5 file containing the SLUG output automatically, and is available through the ``slug_reader``, by doing

   .. code-block:: python

      sim_result.cluster_cloudy[keyword]   # or sim_result.galaxy_cloudy[keyword]

   where ``keyword`` is the output one wishes to access -- see :ref:`sec-output` for a full list of options.

Primary outputs from a ``run_cloudy`` run, which are accessible as keywords, include:

* ``wl``: the wavelength grid on which output spectra are stored
* ``spec_inc``, ``spec_trans``, ``spec_emit``, ``spec_trans_emit``: the incident, transmitted, emitted, and transmitted plus emitted spectra calculated by cloudy, on wavelength grid ``wl``. See cloudy's documentation for precise definitions of each of these radiation fields.
* ``line_wl`` and  ``line_label``: the wavelength and label for each emission line; labels name the species responsible for producing the line, e.g., NII or OIII
* ``line_lum`` the luminosity for each emission line.

By default the ``run_cloudy`` command will run cloudy on every cluster or
galaxy spectrum found in the SLUG output, but it is possible to run only on select
spectra: see the full documentation of ``slug_reader.run_cloudy`` in
:ref:`sec-slugpy-full`. The function also provides many other keywords to control
the behavior of the cloudy run.

Controlling Nebular Properties
-------------------------------

Computing the nebular emission requires specifying the physical
properties of the interstellar gas into which the
photons propagate. Codes like cloudy require that the HII region be
described by an inner radius :math:`r_0` and a number density
:math:`n_0` of hydrogen nuclei at that radius. However,
these parameters are not necessarily the most convenient or descriptive
ones with which to characterize HII regions. For this reason, SLUG
allows users to specify HII region properties in a number of other
more convenient ways.

The basic assumptions made in SLUG's parameterization are
that the HII region is isobaric and isothermal, at all points hydrogen
is fully ionized and helium is singly ionized, and that radiation
pressure is negligible. (Important note: these are the assumptions
used in SLUG's way of writing out the parameters, and they are
approximately true for most HII regions. However, they are *not*
exactly true for the final cloudy calculation, where in general the
temperature is not constant, the ionization states of hydrogen and
helium vary through the nebula, and radiation pressure may or may not
be important.) The HII region occupies a spherical shell bounded by an
inner radius :math:`r_0` and an outer radius :math:`r_1`. The inner
radius is set by the presence of a bubble of shocked stellar wind
material at a temperature :math:`\sim 10^6` K, which is assumed to be
optically thin to ionizing photons. The outer radius is set by the
location where all the ionizing photons have been absorbed.

Under these assumptions, the inner density :math:`n_0` is simply the
(uniform) density :math:`n_{\mathrm{II}}` throughout the ionized
region, and the ionizing photon luminosity passing through a shell of
material at a distance :math:`r` from the stars is 

.. math:: Q(r) = Q(\mathrm{H}^0)
	  \left[1 - \left(\frac{r}{r_S}\right)^3 +
	  \left(\frac{r_0}{r_S}\right)^3\right],

where :math:`Q(\mathrm{H}^0)` is the hydrogen-ionizing luminosity of
the source and :math:`r_S` is the Stromgren radius, given by

.. math:: r_S = \left(\frac{3 Q(\mathrm{H}^0)}{4\pi
	  \alpha_B f_e n_{\mathrm{II}}^2}\right)^{1/3}.

Here :math:`\alpha_B` is the case B recombination coefficient and
:math:`f_e` is the abundance of electrons per H nucleus. For the
purposes of cloudy_slug, we take these two quantities to have
the fixed values :math:`\alpha_B = 2.59\times
10^{-13}\;\mathrm{cm}^3\;\mathrm{s}^{-1}`, appropriate for a
temperature of :math:`10^4` K, and :math:`f_e = 1.1`, appropriate for
a region where He is singly ionized.

From this setup one can define some useful dimensionless numbers. One
is the wind parameter :math:`\Omega` introduced by `Yeh &
Matzner (2012, ApJ, 757, 108)
<http://adsabs.harvard.edu/abs/2012ApJ...757..108Y>`_, which under the
simple assumptions made in cloudy_slug is given by

.. math:: \Omega = \frac{r_0^3}{r_1^3-r_0^3}

i.e., it is just the ratio of the volume occupied by the wind gas to
that occupied by the photoionized gas. The value of :math:`\Omega`
determines whether winds are important (:math:`\Omega \gg 1`) or
unimportant (:math:`\Omega \ll 1`) for the dynamics of the HII
region. The second dimensionless parameter is the volume-averaged
ionization parameter

.. math:: \mathcal{U} = \frac{3}{4\pi (r_1^3-r_0^3)} \int_{r_0}^{r_1}
	  \left(\frac{Q(r)}{4\pi r^2 c f_i n_{\mathrm{II}}}\right)
	  4\pi r^2 \, dr.

Here :math:`f_i` is the number of free ions per H nucleus, and is
equal to :math:`f_i = 1.1` under the assumption that He is singly
ionized. The quantity in parentheses is the ratio of the ionizing
photon to ion number densities at radius :math:`r`. The value of
:math:`\mathcal{U}` is, together with :math:`n_{\mathrm{II}}`, the
most important factor in determining the output spectrum. A third
useful dimensionless parameter is the ionization parameter at the
inner radius,

.. math:: \mathcal{U}_0 = \frac{Q(\mathrm{H}^0)}
	  {4\pi r_0^2 f_i n_{\mathrm{II}} c}.

The various quantities are not unrelated. It is straightforward to
show that they are constrained by the following relationships:

.. math:: r_0 & = \Omega^{1/3} r_S \\

	  r_1 & = \left(1 + \Omega\right)^{1/3} r_S \\

	  \mathcal{U} & = \left[\frac{81 \alpha_B^2 n_{\mathrm{II}}
	  Q(\mathrm{H}^0)}{256 \pi c^3 f_e}\right]^{1/3}
	  \left[\left(1 + \Omega\right)^{4/3} 
	  - \Omega^{1/3} \left(\frac{4}{3}+\Omega\right)\right] \\

	  & = \left[\frac{81 \alpha_B Q(\mathrm{H}^0)}
	  {64 \pi c^2 f_e r_S}\right]^{1/2}
	  \left[\left(1 + \Omega\right)^{4/3} 
	  - \Omega^{1/3} \left(\frac{4}{3}+\Omega\right)\right] \\

	  \mathcal{U}_0 &= \left[
	  \frac{\alpha_B^2 n_{\mathrm{II}} Q(\mathrm{H}^0)}
	  {36 \pi c^3 f_e}\right]^{1/3} \frac{1}{\Omega^{2/3}} \\

	  &= \frac{4}{9}\Omega^{-2/3} \left[(1+\Omega)^{4/3} -
	  \Omega^{1/3}\left(\frac{4}{3}+\Omega\right)\right]^{-1}
	  \mathcal{U} \\

These relations may be used to compute any four of the quantities
:math:`n_{\mathrm{II}}`, :math:`r_0`, :math:`r_1`, :math:`\mathcal{U}`,
:math:`\mathcal{U}_0` and :math:`\Omega` given the other two.

Given this background, ``run_cloudy`` allows the user to specify the
physical properties of the HII region by setting any two of the
following six quantities:

#. The photoionized gas density :math:`n_{\mathrm{II}}`.
#. The inner radius :math:`r_0`.
#. The outer radius :math:`r_1`.
#. The volume-averaged ionization parameter :math:`\mathcal{U}`.
#. The inner radius ionization parameter :math:`\mathcal{U}_0`.
#. The wind parameter :math:`\Omega`.

If no parameters are specified, ``run_cloudy`` defaults to
:math:`n_{\mathrm{II}} = 100\,\mathrm{cm}^{-3}` and 
:math:`\mathcal{U} = 10^{-2.5}` as its choices.

A few caveats are in order at this point.

#. Not all combinations of values are realizable. In addition to the
   obvious constraints (e.g., :math:`r_1 > r_0`), there are some
   subtle ones. For example, for any given ionizing luminosity
   :math:`Q(\mathrm{H}^0)` and density :math:`n_{\mathrm{II}}`, the
   value of :math:`\mathcal{U}` is bounded from above. Increasing the
   wind parameter :math:`\Omega` can allow arbitrarily small values of
   :math:`\mathcal{U}`, but not arbitrarily large ones. If the user
   requests a physically impossible combination of parameters,
   ``run_cloudy`` will either raise an error or issue a warning and
   coerce the values to allowed ones, depending on the options
   selected when calling it.
#. Even for parameters that are not physically impossible, the results
   may not be sensible, and may cause cloudy to crash in extreme
   cases. For example, if one sets :math:`\Omega = 0` and
   :math:`\mathcal{U} = 10^{-4}`, then for an ionizing luminosity of
   :math:`Q(\mathrm{H}^0) = 10^{50}` photons/s (typical for a cluster
   of :math:`\sim 10^4M_\odot`), the corresponding density is
   :math:`n_{\mathrm{II}} \approx 10^{-5}\mbox{ cm}^{-3}`! At this
   density the gas will be fully ionized by cosmic rays and the
   extragalactic background, and it makes no sense to think of it as
   an HII region. Caution is required.
#. The parameter combinations :math:`(r_0,\mathcal{U})` and
   :math:`(r_1,\mathcal{U}_0)` are not allowed
   because they do not define a unique solution for the other
   parameters (the resulting equations have multiple physically-valid
   solutions).
#. The relations given above are only valid if radiation pressure is
   not dynamically significant. If it is, then there are no known
   analytic relations between the various quantities. The 
   code will still run, and will use the relations above, but the
   actual HII region properties may be markedly different from those
   requested. 


.. _ssec-cloudy-slug-template:

The Cloudy Template File
------------------------

Calculations using cloudy are controlled by an input deck, much like SLUG
is. The ``slug_reader.run_cloudy`` routine requires a template to specify
all the cloudy options other than the ionizing spectrum, the starting
density and inner radius, and the abundance scaling, all of which are
automatically added to the template by ``run_cloudy``. If no template is
provided, the code uses the default supplied in
``data/cloudy/cloudy.in_grid_template``. The options in that template
are::

   abundances HII region
   sphere expanding
   constant density
   iterate to convergence
   cosmic rays background
   cmb redshift 0
   stop temperature 1000
   stop efrac 0.1
   save last continuum "$OUTPUT_FILENAME.con"
   save last line array units Angstrom "$OUTPUT_FILENAME.linearr" "LineList_HII.dat"

This template chooses cloudy's default abundance set for an HII region (with
an additional scaling to non-Solar metallicities that is added based on the
metallicity of the cluster or galaxy whose spectrum is being used as input),
expanding spherical geometry, constant density, and a background of cosmic
rays and the cosmic microwave background at redshift 0. See cloudy's documentation
for a full explanation of all the remaining fields.