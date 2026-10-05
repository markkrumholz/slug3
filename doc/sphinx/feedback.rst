.. highlight:: rest

.. _sec-feedback:

Feedback
========

SLUG's Model for Feedback
-------------------------

SLUG can calculate feedback from stellar populations in the form of supernovae
and stellar winds. For supernovae, SLUG calculates the cumulative number of
core-collapse supernovae that have occurred in a stellar population up to a
specified time. For stellar winds, it calculates the instantaneous mass flux,
momentum flux, and energy flux from the stellar winds. Both the supernova and
wind feedback calculations are carried out for both individual stars that are
treated stochastically and for non-stochastic stellar populations. These
calculations are controlled by the choice of supernova and wind feedback model;
we outline these choices and how they are implemented below. All of these choices
are controlled by the :ref:`ssec-parameters-feedback` in :ref:`sec-parameters`.

Supernova Feedback
------------------

Calculation of supernova feedback in SLUG is determined by the range of stellar
initial masses that are assumed to end their lives in core-collapse supernovae.
In calculations where supernova yields are enabled (see :ref:`sec-yields`), this
mass range is by default set by the choice of supernova yield model, so that
yields and supernova counts are computed self-consistently; SLUG properly accounts
for "failed supernovae" that appear as mass gaps in the supernova yield tables.
Alternately, users can specify a fixed range of initial masses that are assumed
to produce supernovae at the ends of their lives. See :ref:`ssec-parameters-feedback`
for details on how to specify this mass range. If neither a yield model nor a
mass range is supplied, the number of supernovae will be set to zero.

Regardless of the choice of model, the number of supernovae computed for a
cluster (a simple stellar population) is determined as the sum of the number of
individually-drawn, stochastic stars that have reached the ends of their lives
and have masses in the range that produces supernovae, plus the contribution
from any part of the stellar population being handled non-stochastically; the
latter is computed as

.. math::

    N_{\rm SN, non-stoch}(t) = \frac{M_*}{\left\langle m\right\rangle} \int_0^{m_{\rm stoch,min}} \frac{dn}{dm} \Theta(t - t_{\rm life}(m, Z_\mathrm{Fe})) I_{\rm SN}(m, Z_\mathrm{Fe}) \, dm,

where :math:`M_*` is the total mass of the
stellar population, :math:`\langle m\rangle` is the mean stellar mass for the
chosen initial mass function, :math:`m_{\rm stoch,min}` is the minimum stellar mass
that is treated stochastically, :math:`dn/dm` is the stellar initial mass function,
:math:`t_{\rm life}(m, Z_\mathrm{Fe})` is the lifetime of a star of mass :math:`m`
and iron metallicity :math:`Z_\mathrm{Fe} = 10^{\rm [Fe/H]}`, :math:`Z_\mathrm{Fe}` is the
iron metallicity of the cluster, and
:math:`I_{\rm SN}(m, Z_\mathrm{Fe})` is an indicator function that
is equal to unity if a star with mass :math:`m` and iron metallicity :math:`Z_\mathrm{Fe}` is in the
mass range that is assumed to produce supernovae, and zero otherwise.
Note that :math:`N_{\rm SN,non-stoch}` need not be integer-valued, and so the
total number of supernovae can be non-integer as well.

For galaxy simulations where there is a continuous star formation history
:math:`\dot{M}_*(t)`, the cumulative number of supernovae is the sum
of the number that occur in all clusters, the number that occur in field stars
that reach the ends of their lives and are in the mass range that produces
supernovae (see :ref:`ssec-parameters-feedback`), and the number that occur
in the non-stochastic, non-clustered part of the population. This latter
contribution is given by

.. math::

    \begin{split}
        N_{\rm SN, non-stoch}(t) =& (1 - f_{\rm cluster}) \int_0^t \frac{\dot{M}_*(t')}{\left\langle m\right\rangle} \int_0^{m_{\rm stoch,min}} \frac{dn}{dm} \\
         & \qquad \int_0^{\infty} \frac{dp}{dZ_\mathrm{Fe}} \Theta(t - t' - t_{\rm life}(m, Z_\mathrm{Fe})) I_{\rm SN}(m, Z_\mathrm{Fe}) \, dZ_\mathrm{Fe} \, dm \, dt',
    \end{split}

where :math:`f_{\rm cluster}` is the fraction of stars that are formed in clusters
and :math:`dp/dZ_\mathrm{Fe}` is the distribution of iron metallicities.

Wind Feedback
-------------

A wind feedback model is a model that specifies the wind velocity for a star of a
given stellar mass, metallicity, and evolutionary stage; note that the wind mass
flux is *not* part of the wind feedback model, since it is determined by the
:ref:`sec-tracks`. The wind momentum and energy fluxes for any given
star are computed as :math:`\dot{p} = \dot{m}_w v_w` and
:math:`\dot{E} = (1/2) \dot{m}_w v_w^2`, where :math:`\dot{m}_w` is the mass flux
determined from the stellar tracks, and :math:`v_w` is the wind velocity determined
from the wind model. Given a choice of wind model, the total wind momentum and
energy fluxes are computed by summing the contributions from all stochastic stars
in the population, plus the contribution from the non-stochastic part of the population.
For cluster simulations the non-stochastic contribution is

.. math::

    (\dot{p}, \dot{E})_{\rm non-stoch}(t) = \frac{M_*}{\left\langle m\right\rangle} \int_0^{m_{\rm stoch,min}} \frac{dn}{dm} \dot{m}_w(m, t, Z_\mathrm{Fe}) \left(v_w(m,t,Z_\mathrm{Fe}), \frac{1}{2} v_w^2(m,t,Z_\mathrm{Fe})\right) \, dm,

where :math:`\dot{m}_w(m,t,Z_\mathrm{Fe})` and :math:`v_w(m,t,Z_\mathrm{Fe})` are
the wind mass flux and velocity for a star of initial mass :math:`m`, age :math:`t`, and
iron metallicity :math:`Z_\mathrm{Fe}`, and :math:`Z_\mathrm{Fe}` is the iron metallicity of the cluster. We implicitly take :math:`\dot{m}_w = 0` for stars that are no
longer alive.

For galaxy simulations, the wind feedback is the sum of the contributions from all clusters,
all field stars, and the non-stochastic part of the population, with the latter computed as

.. math::

    \begin{split}
        (\dot{p}, \dot{E})_{\rm non-stoch}(t) =& (1 - f_{\rm cluster}) \int_0^t \frac{\dot{M}_*(t')}{\left\langle m\right\rangle} \int_0^{m_{\rm stoch,min}} \frac{dn}{dm} \int_0^{\infty} \frac{dp}{dZ_\mathrm{Fe}} \dot{m}_w(m, t - t', Z_\mathrm{Fe}) \\
        & \qquad \left(v_w(m,t - t', Z_\mathrm{Fe}), \frac{1}{2} v_w^2(m,t - t', Z_\mathrm{Fe})\right) \, dZ_\mathrm{Fe} \, dm \, dt'.
    \end{split}

For the purposes of computing wind feedback, SLUG divides stars into four categories:
Wolf-Rayet stars, OB stars, RGB/AGB stars, and all other stars. Classification of stars
into the WR category follows the same convention used for calculating stellar spectra if
a WR spectral library is loaded (see :ref:`sec-atmospheres`). For stars that are not
classified as WR stars, SLUG treats any star with an effective temperature
:math:`>1.1\times 10^4` K as an OB star, any remaining star with a surface gravity
:math:`\log_{10} g < 3.5` (in cgs units) as an RGB/AGB star, and all other stars as
"other". For each category it provides the following options for the wind velocity model:

* Wolf-Rayet stars (``feedback.wr_winds``):

  * ``nugis_lamers_00`` (default): the empirical prescription of `Nugis & Lamers (2000) <https://ui.adsabs.harvard.edu/abs/2000A%26A...360..227N/abstract>`_, which gives the wind velocity as a multiple of the star's escape speed (reduced by electron-scattering radiation pressure) that depends on luminosity and surface composition, with separate calibrations for nitrogen-sequence (WN) and carbon-sequence (WC) stars. Since Nugis & Lamers provide no calibration for hydrogen-rich WNL stars, SLUG applies the WN calibration to them as well. The resulting velocity is clamped to the range :math:`740 - 5500` km/s spanned by the Nugis & Lamers calibration sample, and stars that exceed the electron-scattering Eddington limit are assigned the lower limit of this range.
  * ``l_over_c``: the wind velocity is set to :math:`v_w = L / (\dot{m}_w c)`, the velocity at which the wind momentum flux equals the momentum flux of the star's radiation field, as expected for a wind driven by single scattering of photons. This is the same velocity SLUG uses to compute the transformed radius of WR stars when computing their spectra (see :ref:`sec-atmospheres`).
  * ``none``: Wolf-Rayet stars have zero wind velocity, and thus zero wind momentum and energy flux.

* OB stars (``feedback.ob_winds``):

  * ``vink_sander_21`` (default): the fits of `Vink & Sander (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.2051V/abstract>`_ for the wind velocity as a function of luminosity, effective temperature, and metallicity, using separate fits on the hot and cool sides of the bistability jump. The jump is placed at the metallicity-dependent temperature given by `Vink, de Koter, & Lamers (2001) <https://ui.adsabs.harvard.edu/abs/2001A%26A...369..574V/abstract>`_, so that each of the two fits is extended slightly beyond its range of validity (:math:`T_{\rm eff} \leq 20` kK for the cool-side fit, :math:`T_{\rm eff} \geq 25` kK for the hot-side fit) to cover the gap between them.
  * ``vink_01``: the prescription of `Vink, de Koter, & Lamers (2001) <https://ui.adsabs.harvard.edu/abs/2001A%26A...369..574V/abstract>`_, in which the wind velocity is 2.6 or 1.3 times the star's effective escape speed (reduced by electron-scattering radiation pressure) on the hot or cool side of the bistability jump, respectively, scaled with metallicity as :math:`Z_\mathrm{Fe}^{0.13}`. Stars at or above the electron-scattering Eddington limit have zero wind velocity in this model.
  * ``none``: OB stars have zero wind velocity.

* RGB/AGB stars (``feedback.agb_winds``):

  * ``slug2`` (default): the prescription used in SLUG version 2, which uses the scaling :math:`v_w \propto L^{1/4} Z_\mathrm{Fe}^{1/2}` expected for dust-driven winds from `Elitzur & Ivezić (2001) <https://ui.adsabs.harvard.edu/abs/2001MNRAS.327..403E/abstract>`_, normalized to :math:`v_w = 9.4` km/s at :math:`L = 10^4\,L_\odot` and Solar metallicity, following the empirical calibration of `Goldman et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.465..403G/abstract>`_. Note that the metallicity scaling is likely to be inaccurate for metal-poor stars, whose winds are not primarily dust-driven.
  * ``none``: RGB/AGB stars have zero wind velocity.

* Other stars (``feedback.other_winds``):

  * ``vesc`` (default): the wind velocity is equal to the star's surface escape speed, :math:`v_w = \sqrt{2 G M / R}`, where :math:`M` is the star's current mass and :math:`R` is its radius, computed from its luminosity and effective temperature.
  * ``none``: other stars have zero wind velocity.

In all of the above models that depend on metallicity, the metallicity relative to Solar is taken to be :math:`Z_\mathrm{Fe} = 10^{\rm [Fe/H]}`.
