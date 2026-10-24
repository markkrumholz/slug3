.. highlight:: rest

.. _sec-cluster-slug:

cluster_slug: Inference for Star Clusters
=========================================

SLUG can use a library of cluster simulations to perform Bayesian inference of the
physical properties of star clusters from their observed photometry. The basic tool 
for this is the ``cluster_slug`` class in :ref:`sec-slugpy`.

.. _sec-cluster-slug-quickstart:

Using ``cluster_slug``
----------------------

``cluster_slug`` is a class that is accessible via the :ref:`Python API <sec-slugpy-full>`. It works
following the general :ref:`Bayesian inference <sec-bayesian>` method implemented in SLUG.
The basic steps required to perform inference are as follows:

#. Decide on a library of SLUG simulation outputs to use to perform the inference. This library defines the choice of IMF, stellar tracks, stellar atmospheres, etc., and is produced by :ref:`sec-running`. A :ref:`default library <ssec-cluster-slug-default-library>` is provided for users who do not want to produce their own; see :ref:`ssec-cluster-slug-libraries` for requirements for a valid library if you want to make your own.

#. Create a ``cluster_slug`` object by loading the library, by doing

   .. code-block:: python

       from slugpy import cluster_slug
       cs = cluster_slug('libname')

   where ``'libname'`` is the name of the library to load. To use the default library, instead do
   
   .. code-block:: python

       from slugpy import cluster_slug
       cs = cluster_slug()

   Users can optionally also specify the following keywords in the constructor:

   * ``bw_phot`` and ``bw_phys``: these control the bandwidths used for physical and photometric quantities in the kernel; see :ref:`ssec-cluster-slug-bandwidth` for details. Default bandwidths are 0.1 dex for physical quantities (0.1 mag for ``A_V``), and 0.1 mag for photometric quantities, with no correlation.
   * ``inference_quantities``: the list of quantities to infer from photometry; defaults to whichever of ``["mass", "age", "A_V", "metallicity"]`` vary in the library, but users can choose to infer only a subset of these, and can also choose to infer photometric quantities that have not been observed (e.g., inferring the distribution of ionizing luminosity ``Q(HI)`` from longer-wavelength bands). Mass and age are automatically logged, so for example ``mass`` gives :math:`\log_{10} M` in units of :math:`M_\odot` as the inferred quantity, while ``A_V`` (in mag) and ``metallicity`` (:math:`[\mathrm{Fe/H}]`, already logarithmic) are used as-is.
   * ``priors`` and ``pobs``: the prior probability and the probability of observational detection for each cluster in the library; see :ref:`ssec-cluster-slug-priors-pobs` for details.

#. Add a set of photometric filters on which to perform inference, by doing

   .. code-block:: python

       cs.add_filterset(['filter1', 'filter2', ...])

   where ``'filter1'``, ``'filter2'``, ... are the names of :ref:`photometric filters <sec-phot>`
   available in the library. A ``cluster_slug`` object can have multiple filter sets loaded at
   once; just call ``add_filterset`` as many times as needed.

#. Supply a set of observed photometry, along with an optional set of photometric uncertainties, by doing

   .. code-block:: python

       results = cs.conditional(phot, errors=photerr)

   or, if more than one filter set was loaded at the previous step,

   .. code-block:: python

       results = cs.conditional(phot, errors=photerr, filterset=fs)

   Here ``fs`` is a list of filters specifying which filter set the photometry is for (not
   needed if only a single filter set is loaded, but otherwise required), ``phot`` is an array of
   magnitudes or log luminosities (depending on the photometric system) and ``photerr`` is an array
   (or ``astropy.units.Quantity``) of errors, both of length ``nfilter``, where ``nfilter`` is the
   number of filters in the filterset added in the previous step. The returned value ``results``
   will be a ``KDE`` object giving the PDF of cluster physical properties conditional on the input
   photometry and photometric errors. Alternatively, ``phot`` and ``photerr`` can be arrays of
   shape ``(M,nfilter)``, in which case the function will return a list of ``KDE`` objects of
   length ``M``, one per row of ``phot``; these will be computed in parallel. The input ``phot``
   and ``photerr`` must be in the same photometric system as used in the library; use 
   ``phot_convert`` (:ref:`sec-slugpy`) if necessary to convert observations into the photometric
   system used in the library. For photometric systems using magnitudes, the input ``phot``
   must be absolute magnitudes, not apparent ones.

#. The ``KDE`` objects in results can be used to compute the full PDF of physical properties at any given set of physical parameters by doing

   .. code-block:: python

       results.pdf(x)

   where ``x`` is an array with as many elements as the number of quantities in
   ``inference_quantities``. Alternatively, the KDEs can be further reduced to produce marginal
   1D KDEs for each physical property by doing

   .. code-block:: python

       marginal_kde = results.marginalize(dims)

   where ``dims`` is a list of integers or strings specifying which dimensions or quantities
   in ``inference_quantities`` are to be marginalized out. Thus for example to get the marginal
   posterior PDF of log cluster mass for the default list of ``inference_quantities``, one would
   use ``dims = ["age", "A_V", "metallicity"]``. Alternatively, the KDEs can also be used to
   directly compute confidence intervals on each of the ``inference_quantities`` by doing

   .. code-block:: python

       results.percentile(qty, pct)

   where ``qty`` is a string specifying the quantity whose confidence intervals are to be computed
   (e.g., ``"mass"``) and ``pct`` gives the percentile values to be computed
   (e.g., ``[16, 50, 84]`` to get the 16th, 50th, and 84th percentiles). This function behaves
   analogously to ``np.percentile``.

#. Instead of producing ``KDE`` objects directly, users can also use the ``conditional_sample`` function, by doing

   .. code-block:: python

       pts, wgts, idx = cs.conditional_sample(phot, errors=photerr, filterset=fs)

   This returns the sample points, weights, and indices from which the ``KDE`` is constructed;
   here ``pts`` is an array of shape ``(npts, nqty)``, where ``npts`` is the number of points
   and ``nqty`` is the number of quantities in ``inference_quantities``, ``wgts`` is an array
   of shape ``npts`` containing the weights of each of those points, and ``idx`` is an array
   of integers of shape ``npts`` containing the indices of these points in the library. This
   form is more convenient for writing outputs to file, and the corresponding ``KDE`` objects
   can then be constructed from the points whenever needed by passing ``pts`` and ``wgts``
   to the ``KDE`` class's constructor, along with the desired bandwidth for inference of
   physical quantities (see :ref:`sec-slugpy-full`).

#. A quick way of checking whether a given set of input photometry is reproduced well by the library is to use the ``neighbors`` function to find the nearest neighbors to a given cluster's photometry in the library, by doing

   .. code-block:: python

       dist, idx = cs.neighbors(phot, n, errors=photerr)

   where ``n`` is the number of neighbors to find. As with ``conditional``, if more than one filter
   set is loaded, then the ``filterset`` keyword is also required to specify which filter set
   to use. The return values ``dist`` and ``idx`` are the distances and indices in the library
   of the ``n`` nearest neighbors of the input cluster. Clusters that have at least a few neighbors
   with distances :math:`<2` can be regarded as well-fit by the library, while clusters for which
   the nearest neighbor has a distance :math:`\gtrsim 4` are probably not well-fit by the library. However,
   there is no strict goodness of fit test for a single cluster with a probabilistic model, so
   this is only a rule of thumb.

This is only a partial list of the capabilities of ``cluster_slug``; see :ref:`sec-slugpy-full` for
full documentation of all its methods and properties.

.. _ssec-cluster-slug-default-library:

The Default cluster_slug Library
--------------------------------

Inference requires a library, which is the output of a SLUG cluster-type simulation (see
:ref:`ssec-cluster-vs-galaxy`). A default library is available, which can be obtained by
the following methods depending on how you installed SLUG:

For installs from ``pip`` / PyPI:

.. code-block:: bash

    python -m slugpy.download_data --subdir cluster_slug

For installs using a Docker container:

.. code-block:: bash

    docker run --rm -v /path/to/slug-data:/data ghcr.io/markkrumholz/slug3:latest \
        python -m slugpy.download_data --subdir cluster_slug

For installs using an Apptainer/Singularity container:

.. code-block:: bash

    apptainer exec --bind /path/to/slug-data:/data slug3.sif python -m slugpy.download_data \
        --subdir cluster_slug

If you have cloned the repository and are building from scratch, execute the following from the repository root:

.. code-block:: bash

    python3 data/tools/download_data.py --subdir cluster_slug

**WARNING:** the default library is 33 GB in size because it contains a large sample of clusters
for a wide range of filters. The large size is why it is not included as part of the default
SLUG data download.

The default library uses the following settings:

.. list-table::
   :widths: 20 80

   * - IMF
     - Chabrier (2005): lognormal below :math:`1\,\mathrm{M}_\odot` (centered on :math:`0.2\,\mathrm{M}_\odot`, dispersion 0.55 dex), power law of slope -2.35 from :math:`1-120\,\mathrm{M}_\odot`; full mass range :math:`0.08-120\,\mathrm{M}_\odot`
   * - Stellar tracks
     - MIST, rotating at :math:`v/v_\mathrm{crit} = 0.4`, :math:`[\alpha/\mathrm{Fe}] = 0`
   * - Metallicity
     - [Fe/H] drawn uniformly from -1.0 to +0.3
   * - Cluster mass (CMF)
     - :math:`dN/dM \propto M^{-1}` from :math:`10^2` to :math:`10^5\,\mathrm{M}_\odot`, and :math:`\propto M^{-2}` from :math:`10^5` to :math:`10^7\,\mathrm{M}_\odot`
   * - Cluster age
     - :math:`dN/dt \propto t^{-1}` (uniform in :math:`\log t`) from :math:`10^5 - 1.5 \times 10^{10}` yr
   * - Extinction
     - :math:`A_V` drawn uniformly from 0 to 3 mag, Calzetti starburst extinction curve
   * - Nebular emission
     - Included. The nebular emission's :math:`A_V` is the stellar :math:`A_V` times a factor drawn from a normal distribution of mean 2.1 and dispersion 0.5, truncated to the range 1 - 100.
   * - Spectral synthesis
     - Default spectral model, 1024-point wavelength grid from 200 Å to 20 μm
   * - Photometric system
     - Vega magnitudes

It includes the following filters:

- **HST WFC3/UVIS:** F225W, F275W, F336W, F438W, F555W, F814W, F547M,
  F657N, F689M (``HST.WFC3_UVIS1.*``)
- **HST ACS/WFC:** F435W, F555W, F606W, F814W, F658N (``HST.ACS_WFC.*``)
- **JWST NIRCam:** F115W, F150W, F187N, F200W, F277W, F300M, F335M,
  F405N, F444W (``JWST.NIRCam.*``)
- **JWST MIRI:** F560W, F770W (``JWST.MIRI.*``)
- **Ionizing photon rates:** ``Q(HI)``, ``Q(HeI)``, ``Q(OII)`` (photons/s
  above the H, He, and :math:`\mathrm{O}^+` ionization thresholds)
- **Bolometric luminosity:** ``Lbol``

Note that the JWST NIRCam F200W and redder filters do not include extinction, because they
are beyond the range tabulated in the Calzetti starburst extinction curve. Similarly, the
ionizing photon rates do not include extinction.

.. _ssec-cluster-slug-libraries:

Making your own library for cluster_slug
----------------------------------------

If you prefer to use your own library, you can use most cluster simulations. However, the following
rules and recommendations apply:

* The output must include both the ``clusters`` and ``cluster_phot`` groups; see :ref:`sec-output`.
* There can be no more than one output time per cluster, so either set the ``output_times`` keyword in :ref:`sec-parameters` to a PDF (recommended) or a single value.
* In order to infer cluster mass, age, extinction, and metallicity, the ``CMF``, ``output_times``, ``AV``, and ``FeH`` parameters, respectively, in :ref:`sec-parameters` must be set to PDFs rather than single values; any quantity set to a single value will not be available for inference.
* Every filter on which inference is to be performed must be available in the library, so set the ``filters`` keyword in :ref:`sec-parameters` appropriately.
* If the library includes outputs with either :ref:`sec-extinction` or :ref:`sec-nebular`, these will be used by default when performing inference (though this can be overridden by the ``use_extinct`` and ``use_nebular`` keywords when constructing the ``cluster_slug`` object).
* The size of the library, together with the typical photometric errors, will determine the bandwidth it is safe to use and the accuracy of the resulting inference (see :ref:`ssec-cluster-slug-bandwidth`). For photometric uncertainties at the :math:`\approx 0.1` mag level, libraries of :math:`>10^6`, and preferably :math:`\gtrsim 10^7`, clusters are recommended.

.. _ssec-cluster-slug-priors-pobs:

Priors and Observation Probabilities
------------------------------------

The :ref:`Bayesian inference method <sec-bayesian>` used by ``cluster_slug`` starts from a joint
PDF of physical and photometric properties of star clusters, which is formed using a weighted
kernel density estimate from the SLUG library. By default, if the ``priors`` and ``pobs``
properties of ``cluster_slug`` are left as ``None`` (their default), all clusters in the
library are weighted equally; this is equivalent to assuming that all clusters are equally
likely to have been observed, and to adopting a prior probability for Bayesian inference that
is equal to the probability distribution from which the clusters were sampled when the library
was created, e.g., if the library was created with a ``CMF`` keyword describing a mass
distribution :math:`p(M) \propto M^{-2}`, then leaving ``priors`` as ``None`` is equivalent
to performing Bayesian inference assuming this prior probability for cluster masses.

Users, however, may wish to modify ``priors`` to specify a different choice of prior probability,
and to modify ``pobs`` to take into account selection bias in the observations (e.g., that
clusters with lower luminosities are less likely to be detected). The effect of each of these
parameters is to change the weights in the kernel density estimate; the weight used is

.. math::

    w_i \propto P_{\mathrm{obs},i} \frac{p_{\mathrm{prior},i}}{p_{\mathrm{lib},i}}

where :math:`w_i` is the weight assigned to cluster :math:`i` in the library,
:math:`P_{\mathrm{obs},i}` is the probability that this cluster would be detected
(assumed to be unity if ``pobs`` is ``None``), :math:`p_{\mathrm{prior},i}` is the
prior probability density for the properties of cluster :math:`i`, and
:math:`p_{\mathrm{lib},i}` is the probability density for the properties of cluster
:math:`i` evaluated using the PDFs from which the library was constructed, e.g., if
the ``CMF`` and ``output_times`` keywords in :ref:`sec-parameters` specify PDFs
:math:`p_M(M)` and :math:`p_T(T)`, then we will have 
:math:`p_{\mathrm{lib},i} \propto p_M(M_i) p_T(T_i)`, where :math:`M_i` and
:math:`T_i` are the mass and age of cluster :math:`i` in the library. Note that,
even though inference is performed on log mass and log time, the prior (and library
sampling density) are expressed in linear variables, i.e., :math:`p_{\mathrm{prior},i}`
must be the probability per unit mass and per unit age, not per unit log mass and log age.
Setting ``priors = None`` is equivalent to setting
:math:`p_{\mathrm{prior},i} = p_{\mathrm{lib},i}`.
Note that the absolute values of the weights are not meaningful because their sum
is always normalized to unity. See
`Krumholz et al. (2019) <https://ui.adsabs.harvard.edu/#abs/2019MNRAS.482.3550K/abstract>`_
for a derivation of the theoretical underpinnings of this functional form for the
weights.

In ``cluster_slug``, ``priors`` and ``pobs`` are properties of the class (which can also be
set by providing keywords to the constructor), and can be set to either numpy arrays or callables.
If they are set to arrays, these arrays must have the same length as the number of
clusters in the library (available as the ``size`` property of the ``cluster_slug`` object),
and each array element must be a non-negative real number; for ``pobs`` the number must be
in :math:`[0,1]`. If these are set to callables, the callable must accept four arguments, which will contain the
mass, age, visual extinction, and metallicity of every cluster in the library: the first three as
``astropy.units.Quantity`` arrays of length ``size`` (the extinction argument is ``None`` if the
library has no extinction), and the metallicity :math:`[\mathrm{Fe/H}]` as a plain numpy array of
length ``size``. The callable must return as output
an array of shape ``size`` containing the prior or observation probability for each cluster
(which must again be a non-negative real number, with ``pobs`` restricted to :math:`[0,1]`).


.. _ssec-cluster-slug-bandwidth:

Choosing a Bandwidth
--------------------

Bandwidth is a fundamental choice in :ref:`ssec-bayesian-kernel-density`. The bandwidth determines how much
"blurring" is applied to the library clusters when performing inference, and it represents a
tradeoff between noisiness and accuracy -- a bandwidth that is too large will produce inferences
that are blurred out and under-specific, while one that is too small will produce results
dominated by unphysical small-scale noise. There is no general rule for choosing the "right"
bandwidth, and general rules of thumb such as Scott's rule or Silverman's rule generally do
not apply to the types of data upon which ``cluster_slug`` acts, which are highly-structured
and multimodal. Experimental approaches are often needed. However, there are a few principles
which can help in selecting an appropriate bandwidth for your library.

First, you want the bandwidth to be such that most of your observed clusters will have at least
a few library clusters within the kernel defined by the combination of their photometric
uncertainties and the chosen bandwidth. To judge this, ``cluster_slug`` provides the
``neighbors`` method, which can be invoked as

.. code-block:: python

    dist, idx = cs.neighbors(phot, n, errors=photerr)

where ``n`` is the number of neighbors to find and ``phot`` and ``photerr`` are the photometry
and photometric errors, as in the :ref:`directions <sec-cluster-slug-quickstart>` above. This
will return the distances ``dist`` and indices in the library ``idx`` of the ``n`` clusters
that are closest to the input photometry, where closest is judged using the distance metric
defined by the kernel. For a good choice of bandwidth, most observed clusters should have
at least a few neighbors for which ``dist`` is below 2. If this is not the case, you should
likely increase the photometric bandwidth; conversely, if you are finding tens of neighbors
with distances below 2 you may want to decrease your bandwidth to get more precise results.

Second, you want to be able to set a photometric bandwidth that is at least slightly smaller
than :math:`\sqrt{2}\times` your typical photometric uncertainties without violating the
first principle. This is because the bandwidth used in the estimation in the photometric
dimensions is the quadrature sum of :math:`\sqrt{2}\times` photometric errors and the
bandwidth (in the simple case where the bandwidth and errors are both diagonal; the
relationship is more complex if errors or bandwidths are correlated, but the general
principle still applies). Consequently, if you are able to set a bandwidth smaller than the
typical photometric errors, this means your inferences will be limited by the quality of the
data, rather than by the size of your library. If you find that making the photometric bandwidth
small enough to meet this condition leads to you having too few neighbors, you should
probably generate a larger library. The flip side of this point is that, once you have a
library large enough to satisfy this condition, there is little benefit to using an even
larger one that lets you shrink the photometric bandwidth even more, since your inferences
are now limited by the accuracy of the measurements.

Third, choose your physical bandwidth in reasonable proportion to your photometric one.
A difference of 1 mag corresponds to 0.4 dex, so the physical bandwidth in dex should
be reasonably close (within a factor of around two) to the photometric bandwidth in mag.
Choosing bandwidths that are farther apart than that will either excessively blur or
induce artificial noise in the inferred physical quantities.





