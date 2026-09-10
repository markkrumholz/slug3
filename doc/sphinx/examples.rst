.. highlight:: rest

.. _sec-examples:

Examples
========

The repository contains a number of examples under ``examples``. Each of these illustrates
a different feature of SLUG. Examples that are intended to be run from the command line
contain a ``README.md`` file that describes the test and how to run it, while examples
that are intended to be run from Python provide a .py file that is a 
`marimo <https://marimo.io/>`_ notebook, which contains documentation and comments
explaining the example.

The examples currently available are:

* ``examples/cloudy``: demonstrates how to perform cloudy post-processing (see :ref:`sec-cloudy-slug`) on SLUG outputs
* ``examples/clusterlib``: demonstrates using the command line to carry out a large Monte Carlo simulation of star clusters, a simplified version of the library used in `Tang, Grasha, & Krumholz (2024, MNRAS, 532, 4583) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.532.4583T/abstract>`_
* ``examples/QHIFUVstats``: demonstrates how to use SLUG to study the statistics of the ratio of ionizing photons to FUV luminosity, a simplified version of the calculation in `da Silva, Fumagalli, & Krumholz (2014, MNRAS, 444, 3275) <http://adsabs.harvard.edu/abs/2014MNRAS.444.3275D>`_
* ``examples/QHIstat``: demonstrates how to use SLUG to study the distribution of ionizing luminosity per unit mass for star clusters of different masses; a simplified version of the calculation in the appendix of `Kim, Kim, & Ostriker (2016, ApJ, 819, 137) <https://ui.adsabs.harvard.edu/abs/2016ApJ...819..137K/abstract>`_
