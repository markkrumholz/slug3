.. highlight:: rest

.. _sec-pdfs:

Probability Distribution Functions
====================================

SLUG regards many of its inputs -- for example the initial mass function (IMF)
and the metallicity distribution -- as probability distribution functions (PDFs). SLUG
provides a generic file format for specifying PDFs; examples can be found in the
``data/imfs`` and ``examples/*`` directories in the repository.

Mathematical Description of PDFs
--------------------------------

PDFs in SLUG are generically written as functions

.. math:: \frac{dp}{dx} = n_1 f_1(x; x_{1,a}, x_{1,b}) + n_2 f_2(x; x_{2,a}, x_{2,b}) + n_3 f_3(x; x_{3,a}, x_{3,b}) + \cdots,

where :math:`f_i(x; x_{i,a}, x_{i,b})` is non-zero only for
:math:`x \in [x_{i,a}, x_{i,b}]`. The functions :math:`f_i` are simple continuous
functional forms, which we refer to as *segments*. These segments need not be either
continuous or overlapping with each other, i.e., there is no requirement that
:math:`x_{1,b} = x_{2,a}`.

SLUG further differentiates between normalized and unnormalized PDFs. A normalized PDF
is one that satisfies

.. math:: \int_{-\infty}^\infty \frac{dp}{dx} dx = 1

while an unnormalized PDF need not satisfy this condition; for an unnormalized PDF, the
factors :math:`n_i` are arbitrary.

Specifying PDFs
---------------

When a PDF is called for in a parameter file, it can be provided as either a single number
or the name of a PDF descriptor file. A single number is treated as a delta function at
the specified value. A PDF descriptor file is a `TOML <https://toml.io/en/>`_ file that can
be formatted in one of two ways.

"Basic" Format
**************

Files in this format describe PDFs that are normalized and continuous, meaning that their
segments do satisfy :math:`x_{i,b} = x_{i+1,a}`. A basic-format PDF descriptor file begins
with a single toml keyword at the top level, outside of any table::

    breakpoints = [ x_1, x_2, x_3, ... ]

The value of breakpoints is the list of points :math:`x_i` at which the segments of the PDF
join, i.e., the first entry is :math:`x_1 = x_{1,a}`, the second is
:math:`x_2 = x_{1,b} = x_{2,a}`, etc., and the last entry is :math:`x_N = x_{N,b}`. The
entries in ``breakpoints`` must be strictly non-decreasing, and a pair of points :math:`x_i`
and :math:`x_{i+1}` can be equal to one another only if the segment between them is a
delta function (see :ref:`ssec-pdfs-segment-types`).

This is followed by a series of tables ``[segment1]``, ``[segment2]``, etc., that describe
each segment; the number of segments must be one less than the number of breakpoints. Each
segment table must contain the keyword ``type``, which specifies the functional form for
that segment, plus additional keywords that depend on the segment type and specify its
parameters; see :ref:`ssec-pdfs-segment-types` for a list of all the segment types and
their required parameters.

An example of a basic-format PDF descriptor file (``data/imfs/chabrier.toml``) is:

.. code-block:: toml

    ###############################################################
    # This file defines the Chabrier (2005) IMF          
    ###############################################################

    # Breakpoints: mass values where the functional form changes
    # The first and last breakpoint will define the minimum and
    # maximum mass
    breakpoints = [0.08, 1, 120]

    # Definitions of segments between the breakpoints

    # This segment is a lognormal with a mean of log_10 (0.2 Msun) 
    # and dispersion 0.55; the dispersion is in log base 10, not 
    # log base e
    [segment1]
    type = "lognormal"
    mean = 0.2
    disp = 0.55

    # This segment is a powerlaw of slope -2.35
    [segment2]
    type = "powerlaw"
    slope = -2.35

"Advanced" Format
*****************

The "advanced" format must be used to describe any PDF that is unnormalized, or for which
the segments are not continuous. PDF descriptor files in this format begin before any tables
with the keyword

.. code-block:: toml

    format = "advanced"

This is followed by a series of tables ``[segment1]``, ``[segment2]``, etc. describing each
segment. Each of these tables must contain a ``type`` keyword and all the mandatory
additional parameters for that type (see :ref:`ssec-pdfs-segment-types`), and must also
contain the keywords ``min``, ``max``, and ``weight``. The keywords ``min`` and ``max``
define the two endpoints of the segement, i.e., :math:`x_{i,a}` and :math:`x_{i,b}`, and
weight defines the integral of the segment, i.e.,

.. math:: \int_{x_{i,a}}^{x_{i,b}} f_i(x; x_{i,a}, x_{i,b}) \,dx = \mathrm{weight}

An example of an advanced format PDF definition file is:

.. code-block:: toml

    ###############################################################
    # This file defines a PDF that is intended to be a star
    # formation history consisting of a series of Gaussian bursts
    # with a period of 100 Myr and a width of 10 Myr; the
    # normalization is set such that the time-averaged star
    # formation rate is 1e-3 Msun / yr.
    ###############################################################

    format = "advanced"

    # First burst
    [segment1]
    type = "normal"
    min = 0.0       # Start of burst in yr
    max = 1.0e8     # End of burst in yr
    weight = 1e5    # Total stellar mass formed in burst in Msun
    mean = 0.0      # Center of the burst
    disp = 1.0e7    # Dispersion in yr of the burst

    # Second burst
    [segment2]
    type = "exponential"
    min = 1.0e8     # Start of burst in yr
    max = 2.0e8     # End of burst in yr
    weight = 1e5    # Total stellar mass formed in burst in Msun
    mean = 1.0e8    # Center of the burst
    disp = 1.0e7    # Dispersion in yr of the burst

    # Third burst
    [segment3]
    type = "normal"
    min = 2.0e8     # Start of burst in yr
    max = 3.0e8     # End of burst in yr
    weight = 1e5    # Total stellar mass formed in burst in Msun
    mean = 2.0e8    # Center of the burst
    disp = 1.0e7    # Dispersion in yr of the burst

.. _ssec-pdfs-segment-types:

PDF Segment Types
*****************

The following table lists the permitted segments types and their required additional
keywords.

.. _tab-segtypes:

.. table:: Segment Types

   +-----------------+--------------------------------------------------------+-----------+---------------------------+-----------+-------------------------------------------------+
   | Name            | Functional form                                        | Keyword   | Meaning                   | Keyword   | Meaning                                         |
   +=================+========================================================+===========+===========================+===========+=================================================+
   | ``delta``       | :math:`\delta(x-x_a)`                                  |           |                           |           |                                                 |
   +-----------------+--------------------------------------------------------+-----------+---------------------------+-----------+-------------------------------------------------+
   | ``exponential`` | :math:`\exp(-x/x_*)`                                   | ``scale`` | Scale length, :math:`x_*` |           |                                                 |
   +-----------------+--------------------------------------------------------+-----------+---------------------------+-----------+-------------------------------------------------+
   | ``lognormal``   | :math:`x^{-1} \exp\{-[\log_{10}(x/x_0)]^2/2\sigma^2\}` | ``mean``  | Mean, :math:`x_0`         | ``disp``  | Dispersion in :math:`\log_{10}`, :math:`\sigma` |
   +-----------------+--------------------------------------------------------+-----------+---------------------------+-----------+-------------------------------------------------+
   | ``normal``      | :math:`\exp[-(x-x_0)^2/2\sigma^2]`                     | ``mean``  | Mean, :math:`x_0`         | ``disp``  | Dispersion, :math:`\sigma`                      |
   +-----------------+--------------------------------------------------------+-----------+---------------------------+-----------+-------------------------------------------------+
   | ``powerlaw``    | :math:`x^p`                                            | ``slope`` | Slope, :math:`p`          |           |                                                 |
   +-----------------+--------------------------------------------------------+-----------+---------------------------+-----------+-------------------------------------------------+
   | ``schechter``   | :math:`x^p \exp(-x/x_*)`                               | ``slope`` | Slope, :math:`p`          | ``x_star``| Cutoff, :math:`x_*`                             |
   +-----------------+--------------------------------------------------------+-----------+---------------------------+-----------+-------------------------------------------------+

Sampling Methods
****************

Both basic-mode and advanced-mode PDF definition files can contain an additional
keyword ``method`` in their initial section, before any segment tables. The method
keyword describes the sampling method used when drawing samples from the PDF that
are intended to add up to some target value -- for example drawing stars from the
IMF to reach some target total mass of the stellar population. Since an integer
number of random draws from the PDF will in general never hit the exact target, a policy
is required for how to handle mismatches. The following options are available when
drawing from a PDF :math:`dp/dx` where the samples are intended to sum to a target
value :math:`x_\mathrm{target}`:

* ``stop_nearest``: this is the default option: draw until the sum of the values drawn exceeds :math:`x_\mathrm{target}`, then either keep or discard the final draw depending on which choice brings the total closer to the target.
* ``stop_before``: same as ``stop_nearest``, but the final draw is always excluded.
* ``stop_after``: same as ``stop_nearest``, but the final draw is always kept.
* ``stop_50``: same as ``stop_nearest``, but keep or exclude the final draw with 50% probability regardless of which choice gets closer to the target.
* ``number``: draw exactly :math:`N = \operatorname{round}(x_\mathrm{target}/\langle x\rangle)` samples, where :math:`\langle x\rangle = \int_{-\infty}^{\infty} x (dp/dx) \, dx / \int_{-\infty}^{\infty} (dp/dx) \, dx` is the expectation value of the PDF.
* ``poisson``: draw exactly :math:`N` objects, where the value of :math:`N` is chosen from a Poisson distribution with expectation value :math:`\langle N \rangle = x_\mathrm{target}/\langle x\rangle`. Note that this is the *only* sampling policy for which the mean, expectation value, and similar statistical properties of the population drawn are identical to those of the underlying PDF.
* ``sorted_sampling``: this method was introduced by `Weidner & Kroupa (2006, MNRAS. 365, 1333) <http://adsabs.harvard.edu/abs/2006MNRAS.365.1333W>`_, and proceeds in steps. One first draws exactly :math:`N= x_\mathrm{target}/\langle x\rangle` samples as in the ``number`` method. If the resulting total of the draws :math:`x_\mathrm{sum} = \sum_{i=1}^N x_i` is less than :math:`x_\mathrm{target}`, the procedure is repeated recursively using a new target :math:`x_\mathrm{target} - x_\mathrm{sum}` until :math:`x_\mathrm{sum} > x_\mathrm{target}`. Finally, one sorts the resulting draws from smallest to largest, and then keeps or removes the final, largest sample using a ``stop_nearest`` policy.

See ``data/imfs/weidner_kroupa06.toml`` for an example of a PDF definition file that
uses a sampling policy.