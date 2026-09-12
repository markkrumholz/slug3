### Overview of slug3 ###

This is version 3 of the Stochastically Lighting Up Galaxies (SLUG) code. 

SLUG is a stellar population synthesis (SPS) code, meaning that for an input star formation history, stellar initial mass function, and a set of evolutionary tracks and stellar atmospheres it can predict the light output of the stellar population. The main difference between SLUG and conventional SPS codes is that, instead of the usual approach of assuming that all stellar masses and ages are fully populated, SLUG is capable of stochastically sampling from the stellar initial mass function and age distribution, and thereby predicting not just the mean light output but also the full distribution that results from stochastic sampling. This capability is critical in the regime of low star formation rates and total stellar masses, where finite sampling can lead to a distribution of properties that is extremely broad, and the mean values produced by other SPS codes are therefore of limited predictive power.

### Version history ###

This is version 3 of SLUG. It is a complete rewrite of version 2, taking advantage of modern C++ and Python capabilities to significantly expand the usability. It also greatly expands the range of stellar tracks and atmospheres available, taking advantage of advances that have occurred since the last release. Major improvements in this version include:

* The ability to drive SLUG entirely from Python, as well as the traditional command line interface.
* Native support for hybrid MPI + OpenMP parallelism.
* A modernized build system that greatly reduces the number of external dependencies and makes building the code across platforms significantly easier.
* Output has been migrated from the FITS format used in version 2 to [HDF5](https://www.hdfgroup.org/solutions/hdf5/) format, allowing significantly faster and more flexible IO.
* The input format has been simplified and converted to the [TOML](https://toml.io/en/) standard, providing a more intuitive way of controlling the code.
* A larger, modernized set of stellar tracks, atmospheres, and photometric filters, and standardization of the formats for describing these and other data inputs so that users can easily add additional data without needing to alter the source code.

Version 2 of SLUG remains available at <https://bitbucket.org/krumholz/slug2/>. As of this writing there are some capabilities in SLUG version 2 that have not yet been re-implemented in this version, and while this remains true SLUG version 2 will continue to be maintained. However, version 2 is no longer being developed, and eventually all capabilities it provides will be migrated to this version, at which point maintenance on SLUG version 2 will be discontinued. Users are therefore encouraged to migrate to this version.

### Documentation ###

See <https://markkrumholz.github.io/slug3/> for full documentation, including detailed
instructions on compiling and running the code.

### Layout of this repository ###

This repository is structured as follows:

* *.github*: GitHub workflows
* *cmake*: helper CMake build files
* *data*: data files, divided by type of data
    - *data/cloudy*: template files for the SLUG-Cloudy interface
    - *data/elem*: elemental and atomic data
    - *data/filters*: filter response curves and similar data
    - *data/extinct*: extinction curves
    - *data/imfs*: stellar initial mass functions
    - *data/nebular*: tabulated nebular emission
    - *data/spectra*: stellar atmosphere and spectral library data
    - *data/tools*: scripts for fetching and manipulating data
    - *data/tracks*: libraries of stellar evolutionary tracks
* *doc/sphinx*: documentation
* *examples*: examples applications of SLUG, each in a separate subdirectory
* *slugpy*: the Python frontend for SLUG
    - *slugpy/cloudy*: the interface between SLUG and [Cloudy](http://nublado.org/)
* *src*: the main C++ codebase
    - *src/core*: the core C++ classes that run the simulation
    - *src/elem*: elemental and atomic data
    - *src/extern*: external dependencies
    - *src/interpolation*: low-level interpolation machinery
    - *src/io*: input/output management
    - *src/nebular*: nebular emission physics
    - *src/pdfs*: low-level mathematical description of probability distribution functions
    - *src/phot*: photometry and filter calculations
    - *src/pybind*: C++ / Python binding layer
    - *src/specsyn*: spectral synthesis and stellar atmosphere routines
    - *src/tracks*: stellar evolutionary track routines
    - *src/utils*: miscellaneous utilities
* *tests*: unit tests; the subdirectory structure exactly mirrors that of *src*

### Contact ###

If you have questions about SLUG, have discovered any bugs, or want to contribute to ongoing development, please contact [Mark Krumholz](http://www.mso.anu.edu.au/~krumholz/), mark.krumholz@anu.edu.au.
