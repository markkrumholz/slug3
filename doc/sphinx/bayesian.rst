.. highlight:: rest

.. _sec-bayesian:

Bayesian Inference with SLUG
============================

SLUG can be used not only to generate stochastic stellar populations, but also
to use libraries of stochastically-sampled synthetic populations to do Bayesian
inference from observations. The full theory behind SLUG's Bayesian inference
methods is described in `Krumholz et al. (2015) <http://adsabs.harvard.edu/abs/2015MNRAS.452.1447K>`_
and `Krumholz et al. (2019) <https://ui.adsabs.harvard.edu/#abs/2019MNRAS.482.3550K/abstract>`_,
and this page just gives a brief summary of the theory intended for users who
want to implement their own Bayesian inference pipelines.

Physical-Photometric PDFs and Bayesian Inference
------------------------------------------------

SLUG's :ref:`basic operation <sec-intro>` is, given a choice of physical models
for stars (e.g., evolutionary tracks, IMF, atmospheres, etc.), to generate Monte
Carlo realizations of the resulting stellar population, sampled by drawing
stochastically from the IMF and, for :ref:`galaxy simulations <ssec-cluster-vs-galaxy>`,
from the distribution of stellar ages. Each realization produces (among other outputs)
the photometric properties of the cluster drawn. Mathematically, we can describe this
process as conditional sampling from a PDF. We have a space defined by a vector of
physical parameters :math:`\boldsymbol{\Theta}_\mathrm{phys}` (e.g., star cluster
mass, age, extinction, metallicity) and a vector of photometric parameters
:math:`\boldsymbol{\Theta}_\mathrm{phot}` (e.g., magnitudes in various filters),
and we can imagine that there is a joint PDF of physical and photometric properties
:math:`p(\boldsymbol{\Theta})`, where
:math:`\boldsymbol{\Theta} = (\boldsymbol{\Theta}_\mathrm{phys}, \boldsymbol{\Theta}_\mathrm{phot})`
is the vector of physical and photometric properties. The Monte Carlo realizations that
SLUG draws are then distributed in photometry as

.. math:: p_\mathrm{phot}(\boldsymbol{\Theta}_\mathrm{phot}) = \int p(\boldsymbol{\Theta}) \, \mathrm{d}\Theta_\mathrm{phys,1} \ldots \, \mathrm{d}\Theta_\mathrm{phys,N}

where the physical dimensions of :math:` p(\boldsymbol{\Theta})` are distributed following
the PDFs :math:`p_1(\Theta_\mathrm{phys,1}), p_2(\Theta_\mathrm{phys,2}),\ldots` specified
by the input deck (e.g., the PDF of cluster masses, the PDF of metallicities, etc.).

Just as we can regard the SLUG realizations as computing the photometric distribution
marginalized over the physical PDFs, we can equivalently use Bayes' Theorem to write
the PDF of physical variables given a set of measured photometry. Formally, if we
fixed the photometric properties for some observed object to be
:math:`\boldsymbol{\Theta}_\mathrm{phot,obs}`, then we can write the corresponding
PDF of physical properties as

.. math:: p_\mathrm{phys}(\boldsymbol{\Theta}_\mathrm{phys} \mid \boldsymbol{\Theta}_\mathrm{phot,obs}) = \mathcal{N} \int p(\boldsymbol{\Theta}) \delta(\boldsymbol{\Theta}_\mathrm{phot} - \boldsymbol{\Theta}_\mathrm{phot,obs}) \, \mathrm{d}\boldsymbol{\Theta}_\mathrm{phot}

where :math:`\mathcal{N}` is a normalization factor chosen to enforce that the marginalized
PDF over physical properties has unit integral. This gives us a way to infer physical
properties from a set of photometric properties.

The principle challenge then becomes how to construct the continuous joint physical-photometric
PDF :math:`p(\boldsymbol{\Theta})` from the samples provided by a SLUG simulation, and how to
use that estimate to compute the integrals required to return the marginal posterior PDF 
:math:`p_\mathrm{phys}(\boldsymbol{\Theta}_\mathrm{phys} \mid \boldsymbol{\Theta}_\mathrm{phot,obs})`,
in a computationally tractable manner.

.. _ssec-bayesian-kernel-density:

Kernel Density Estimation
-------------------------

SLUG solves this problem by constructing a kernel density estimate from the samples produced
by a SLUG simulation. Consider a SLUG output consisting of :math:`N` trials; we write the
joint physical-photometric PDF as

.. math:: p(\boldsymbol{\Theta}) = \mathcal{N}_\mathbb{G} \sum_{i=1}^{N} w_i \exp(-\mathbf{d}_i^\mathrm{T} \mathbb{G} \mathbf{d}_i),

where :math:`\mathbf{d}_i = \boldsymbol{\Theta} - \boldsymbol{\Theta}_i` is the difference
between a point :math:`\boldsymbol{\Theta}` in the joint physical-photometric space and
the physical-photometric parameters :math:`\boldsymbol{\Theta}_i` of the trial :math:`i`,
:math:`w_i` is the weight assigned to trial :math:`i` (with :math:`\sum_{i=1}^N w_i = 1`),
:math:`\mathbb{G}` is a real symmetric positive-definite matrix that defines the kernel
(which is related to the usual bandwidth matrix :math:`\mathbb{H}` by
:math:`\mathbb{G} = (\mathbb{H}^2)^{-1}`), and :math:`\mathcal{N}_G` is a normalization
constant satisfying
:math:`\mathcal{N}_\mathbb{G} \int \exp(-\mathbf{x}^\mathrm{T} \mathbb{G} \mathbf{x}) \, \mathrm{d}\mathbf{x} = 1`.

This functional form has three main advantages. First, it allows :math:`p(\boldsymbol{\Theta})`
to be evaluated to high accuracy in order :math:`\ln N` time using a tree-based divide-and-conquer
strategy (after an initial, one-time order :math:`N\ln N` operation to construct a tree).

Second, for this functional form, integrals of the form
:math:`\int p(\boldsymbol{\Theta}) \delta(\boldsymbol{\Theta}_\mathrm{phot} - \boldsymbol{\Theta}_\mathrm{phot,obs}) \, \mathrm{d}\boldsymbol{\Theta}_\mathrm{phot}`,
as required to compute the marginal PDF of physical properties given photometric observations,
can also be evaluated in order :\math:`\ln N` time using the same tree-based techniques. The
outcome of such an integral is another kernel density estimate of the same form given by the
equation above, simply over the physical dimensions only.

Third, if the observed photometry :math:`\boldsymbol{\Theta}_\mathrm{phot,obs}` has Gaussian
uncertainties described by a covariance matrix :math:`\bbSigma`, the conditional probability
may be obtained by convolving the expression 
:math:`p_\mathrm{phys}(\boldsymbol{\Theta}_\mathrm{phys} \mid \boldsymbol{\Theta}_\mathrm{phot,obs})`
given above with the uncertainty distribution around :math:`\boldsymbol{\Theta}_\mathrm{phot,obs}`
-- and this convolution can be computed analytically simply by replacing :math:`\mathbb{G}`
with a new bandwidth :math:`\mathbb{G}'` generated by the standard rule for convolution
of Gaussians; consequently, marginalization over an uncertainty distribution is only
marginally more expensive that calculation of the case where the observations are exact.

SLUG's Implementation
---------------------

SLUG provides a class called ``KernelDensity`` -- accessible via :ref:`sec-cpp`, and via
a wrapper class called ``KDE`` via :ref:`sec-slugpy-full` -- to perform kernel density
estimation and computation of conditional and marginal PDFs from it. Convolution of
bandwidths is provided by the ``Bandwidth`` class, also accessible via both APIs. These
classes provide the ability to carry out the mathematical operations described above with
high efficiency and in parallel. The :ref:`cluster_slug <sec-cluster-slug>` cluster
estimation tool is built on this foundation, and can serve as a guide for designing other
Bayesian inference tools.