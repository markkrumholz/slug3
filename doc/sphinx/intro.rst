.. highlight:: rest

Introduction to SLUG
====================

This is a guide for users of the SLUG software package. If you use SLUG
in any published work, please cite the following papers:

* `da Silva, R. L., Fumagalli, M., & Krumholz, M. R., 2012, The Astrophysical Journal, 745, 145 <http://adsabs.harvard.edu/abs/2012ApJ...745..145D>`_
* `Krumholz, M. R., Fumagalli, M., da Silva, R. L., Rendahl, T., & Parra, J. 2015, Monthly Notices of the Royal Astronomical Society, 452, 1447 <http://adsabs.harvard.edu/abs/2015MNRAS.452.1447K>`_

What Does SLUG Do?
------------------

SLUG is a stellar population synthesis (SPS) code, meaning that, for a 
specified stellar initial mass function (IMF), star formation history (SFH), 
cluster mass function (CMF), cluster lifetime function (CLF), metallicity
distribution [Fe/H], and (optionally)  distribution of extinctions (A_V), it
predicts the spectra and photometry of both individual star clusters (treated
as simple stellar populations) and galaxies (composite populations). It can
also predict the yields of various isotopes, and the feedback power provided
by stellar winds and supernovae. In this regard, SLUG operates much like any
other SPS code. The main difference is that SLUG regards the functions describing
the stellar population as probability distributions, and the resulting stellar
population as being the result of a draw from them. SLUG performs a Monte Carlo
simulation to determine the PDF of the outputs produced by the stellar populations
that are drawn from these distributions. The remainder of this section briefly
describes the major conceptual pieces of a SLUG simulation. Please refer to the
papers references above for a full description of the theoretical underpinning of SLUG.

Cluster Simulations and Galaxy Simulations
------------------------------------------

SLUG can simulate either a simple stellar population (i.e., a group of stars all
born at one time) or a composite stellar population, consisting of stars born at
a distribution of times. We refer to the former case as a "cluster" simulation, and
the latter as a "galaxy" simulation, since one can be thought of as approximating
the behavior of a single star cluster, and the other as approximating a whole galaxy. 

Probability Distribution Functions and Monte Carlo Sampling
-----------------------------------------------------------

As mentioned above, SLUG regards many of the characteristics describing a stellar
population -- for example the initial mass function (IMF), the star formation history
(SFH), and the metallicity ([Fe/H]) -- as probability distribution functions. These PDFs can be described by a very
wide range of possible functional forms; see :ref:`sec-pdfs` for details on the
exact functional forms allowed, and on how they can be specified in the code. In its
default operating mode, when SLUG generates a stellar population it draws from these
PDFs to produce a stellar population star-by-star, drawing stellar masses from the
IMF (and, in a galaxy simulation, stellar ages from the PDF associated with the star
formation rate), and similarly drawing other parameters such as the amount of extinction
from their own PDFs. It then calculates the composite spectra, photometry, yields, and
feedback power of the resulting population by using one of several possible sets of
stellar evolutionary tracks and stellar atmosphere models. Since each draw from the PDFs
is random, the resulting stellar population will be different each time SLUG is run. By
running many trials, one can build up a probability distribution function for the
outputs of interest that arises from the stochastic nature of star formation.

However, users also have the option to disable some or all of the stochastic sampling
in SLUG, for example by specifying that the IMF should be fully sampled, or that the
SFH should be treated as a continuous function rather than a probability distribution.
In this case, SLUG will determine the outputs of interest by numerically integrating
over the various distributions, exactly as happens in other, non-stochastic SPS codes.
Simulations can also be "partially stochastic", where some parts of the distributions
are handled stochastically and others deterministically -- for example, it is possible
to do stochastic sampling only for stars above a certain mass. See :ref:`sec-pdfs` for
details on how to control the stochasticity of the various distributions.

SLUG Physics
------------

SLUG includes a number of physics modules that control different aspects of the stellar
population synthesis. The main physics modules are:

* Stellar tracks: SLUG can use a variety of different sets of stellar evolutionarytracks, which
  determine the evolution of stars of different masses and metallicities. See :ref:`sec-tracks` for details.
* Stellar atmospheres: SLUG can use a variety of different sets of stellar atmosphere models, which
  determine the spectra and photometry of stars of different masses, metallicities, and evolutionary stages.
  See :ref:`sec-atmospheres` for details.
* Filters and photometry: SLUG can calculate the photometry of stars and stellar populations
  in a wide range of different filters and using a range of photometric systems. See 
  :ref:`sec-filters` for details.
* Nebular emission: SLUG can calculate the contribution of nebular emission to the spectra
  and photometry of stellar populations, using a variety of different models for the ionized gas. See
  :ref:`sec-nebular` for details.
* Extinction: SLUG can calculate the effects of dust extinction on the spectra and photometry
  of stellar populations, using a variety of different extinction laws. See :ref:`sec-extinction` for details.