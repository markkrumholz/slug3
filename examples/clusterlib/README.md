This example demonstrates an example of a real, large-scale Monte Carlo population
of star clusters -- using MIST tracks, a full chained set of stellar atmosphere models,
spectra and photometry (including dust extinction), and every stochastic input (metallicity, cluster mass, output time, visual extinction) drawn from its own
distribution rather than fixed. Run this from within this directory (so the
dists/*.toml paths below resolve) with:

slug clusterlib.toml
