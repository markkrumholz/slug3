.. highlight:: rest

.. _sec-filters:

Photometry and Filters
======================

Once it has generated a spectrum for a stellar population (see :ref:`sec-atmospheres`),
SLUG can convolve that spectrum with filter response curves to produce predictions for
photometry. The fundamental calculation that SLUG carries out to compute photometry is
to evaluate the integral

.. math:: \left\langle L_\lambda \right\rangle_R = \frac{\int_{-\infty}^\infty L_\lambda R_\lambda \,\mathrm{d}\ln\lambda}{\int_{-\infty}^\infty R_\lambda \,\mathrm{d}\ln\lambda}

where :math:`L_\lambda` is the specific luminosity and :math:`R_\lambda` is the response
function of the filter, which describes the expected number of counts produced in the
detector per incident photon of wavelength :math:`\lambda`. Which filter response
function(s) are evaluated is determined by the ``filters`` keyword
in the :ref:`ssec-parameters-phot` section of :ref:`sec-parameters`. 
See :ref:`ssec-phot-filterdata` and :ref:`ssec-phot-idealfilters` for a list of
available filters in the standard data set, and :ref:`ssec-phot-adding-filters`
for instructions on how to add additional filter data.

Photometric Systems
-------------------

The fundamental integral of the spectrum :math:`L_\lambda` over a filter response
function :math:`R_\lambda` returns a mean specific luminosity over the filter in
the same units as :math:`L_\lambda`: energy per unit time per unit wavelength. SLUG
can automatically convert this output into a variety of other photometric systems,
as specified by the ``system`` keyword in the :ref:`ssec-parameters-phot` section
of :ref:`sec-parameters`. The allowed photometric systems are:

* ``Flambda``: the default, returns :math:`\left\langle L_\lambda \right\rangle_R` directly
* ``Fnu``: returns a specific luminosity :math:`\left\langle L_\nu \right\rangle_R` with units of energy per time per frequency, which is related to :math:`\left\langle L_\lambda \right\rangle_R` by 


.. _ssec-phot-filterdata:

Filter Data
-----------

.. _ssec-phot-idealfilters:

Ideal and Special Filters
-------------------------

.. _ssec-phot-adding-filters:

Adding Additional Filters
-------------------------