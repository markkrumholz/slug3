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

#. Decide on a library of SLUG simulation outputs to use to perform the inference. This library defines the choice of IMF, stellar tracks, stellar atmospheres, etc., and is produced by :ref:`sec-running`. See :ref:`ssec-cluster-slug-libraries` for details on the requirements for a valid library.

#. Create a ``cluster_slug`` object by loading the library, by doing

   .. code-block:: python

       from slugpy import cluster_slug
       cs = cluster_slug('libname')

   Users can optionally also specify the following keywords in the constructor:

   * ``bw_phot`` and ``bw_phys``: these control the bandwidths used for physical and photometric quantities in the kernel; see :ref:`ssec-cluster-slug-bandwidth` for details. Default bandwidths are 0.1 dex for physical quantities (0.1 mag for ``A_V``), and 0.1 mag for photometric quantities, with no correlation.
   * ``inference_quantities``: the list of quantities to infer from photometry; defaults to whichever of ``["mass", "age", "A_V", "metallicity"]`` vary in the library, but users can choose to infer only a subset of these, and can also choose to infer photometric quantities that have not been observed (e.g., inferring the distribution of ionizing luminosity ``Q(HI)`` from longer-wavelength bands). Mass and age are automatically logged, so for example ``mass`` gives :math:`\log_{10} M` in units of :math:`M_\odot` as the inferred quantity, while ``A_V`` (in mag) and ``metallicity`` (:math:`[\mathrm{Fe/H}]`, already logarithmic) are used as-is.
   * ``priors`` and ``pobs``: the prior probability to assign to each combination of physical properties, and the probability that a cluster with a given set of physical or photometric properties will be observed; these can be either callable functions or arrays with as many elements as the number of samples in the library giving a prior and observation probability for each library cluster.

#. Add a set of photometric filters on which to perform inference, by doing

   .. code-block:: python

       cs.add_filterset(['filter1', 'filter2', ...])

   where ``'filter1'``, ``'filter2'``, ... are the names of :ref:`photometric filters <sec-phot>`
   available in the library.

#. Supply a set of observed photometry in that filter set, along with an optional set of photometric uncertainties, by doing

   .. code-block:: python

       results = cs.conditional(phot, errors=photerr)

   where ``phot`` is an array of magnitudes and ``photerr`` is an array (or ``astropy.units.Quantity``) of errors, both of length ``nfilter``, where ``nfilter`` is the number of filters in the filterset added in the previous step. The returned value ``results`` will be a ``KDE`` object giving the PDF of cluster physical properties conditional on the input photometry and photometric errors. Alternatively, ``phot`` and ``photerr`` can be arrays of shape ``(M,nfilter)``, in which case the function will return a list of ``KDE`` objects of length ``M``, one per row of ``phot``; these will be computed in parallel.

#. The ``KDE`` objects in results can be used to compute the full PDF of physical properties at any given set of physical parameters by doing

   .. code-block:: python

       results.pdf(x)

   where ``x`` is an array with as many elements as the number of quantities in ``inference_quantities``. Alternatively, the KDEs can be further reduced to produce marginal 1D KDEs for each physical property by doing

   .. code-block:: python

       marginal_kde = results.marginalize(dims)

   where ``dims`` is a list of integers specifying which dimensions are to be marginalized out. Thus for example to get the marginal posterior PDF of log cluster mass for the default list of ``inference_quantities``, one would use ``dims = [1,2,3]``.

This is only a partial list of the capabilities of ``cluster_slug``; see :ref:`sec-slugpy-full` for
full documentation of all its methods and properties.

.. _ssec-cluster-slug-libraries:

Libraries for cluster_slug
--------------------------

Inference requires a library, which is the output of a SLUG cluster-type simulation (see
:ref:`ssec-cluster-vs-galaxy`). While most cluster simulations can be used, the following
rules and recommendations apply when generating a library.

* The output must include both the ``clusters`` and ``cluster_phot`` groups; see :ref:`sec-output`.
* There can be no more than one output time per cluster, so either set the ``output_times`` keyword in :ref:`sec-parameters` to a PDF (recommended) or a single value.
* In order to infer cluster mass, age, extinction, and metallicity, the ``CMF``, ``output_times``, ``AV``, and ``FeH`` parameters, respectively, in :ref:`sec-parameters` must be set to PDFs rather than single values; any quantity set to a single value will not be available for inference.
* Every filter on which inference is to be performed must be available in the library, so set the ``filters`` keyword in :ref:`sec-parameters` appropriately.
* If the library includes outputs with either :ref:`sec-extinction` or :ref:`sec-nebular`, these will be used by default when performing inference (though this can be overridden by the ``use_extinct`` and ``use_nebular`` keywords when constructing the ``cluster_slug`` object).
* The size of the library, together with the typical photometric errors, will determine the bandwidth it is safe to use and the accuracy of the resulting inference (see :ref:`ssec-cluster-slug-bandwidth`). For photometric uncertainties at the :math:`\approx 0.1` mag level, libraries of :math:`>10^6`, and preferably :math:`\gtrsim 10^7`, clusters are recommended.

.. _ssec-cluster-slug-bandwidth:

Choosing a Bandwidth
--------------------

Bandwidth is a fundamental choice in :ref:`ssec-kernel-density`. The bandwidth determines how much
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





