import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Copyright (c) 2026 Mark Krumholz. All rights reserved.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inferring star cluster masses and ages with `cluster_slug`

    This notebook uses `cluster_slug` to infer the masses and ages of 50 star
    clusters in the galaxy NGC 628 from their HST photometry in five bands,
    taken from the LEGUS star cluster catalog (see the header of
    `legus_ngc628c.txt` for the data's provenance).

    **Requirement:** this notebook uses the default `cluster_slug` library, which
    is *not* downloaded with slug's other data, because of its size (about 33
    GB). Download it before running this notebook; see
    [the instructions in the documentation](https://markkrumholz.github.io/slug3/clusterslug.html#ssec-cluster-slug-default-library).
    For example, for a pip-installed slug, run

    ```
    python -m slugpy.download_data --subdir cluster_slug
    ```

    or, from a checkout of the slug repository, from the repository root,

    ```
    python3 data/tools/download_data.py --subdir cluster_slug
    ```
    """)
    return


@app.cell
def _():
    # Import marimo for markdown rendering
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Add slugpy to path, and import cluster_slug
    import sys
    import os.path as osp
    sys.path = [osp.join('..', '..')] + sys.path
    from slugpy import cluster_slug

    return (cluster_slug,)


@app.cell
def _():
    # Import other tools, including the corner plot routine npdfplot,
    # from this directory
    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    from astropy.io import ascii
    from astropy.table import Table
    from npdfplot import npdfplot

    return Table, ascii, np, npdfplot, plt, u


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The observations

    First we read the photometry and its errors. The catalog gives apparent
    magnitudes, while the library holds absolute magnitudes, so we subtract
    NGC 628's distance modulus of 29.98 mag. The library also names each
    filter by its instrument, so we map the catalog's band names to the
    library's filter names.
    """)
    return


@app.cell
def _(ascii, np):
    # Read the photometry
    obs = ascii.read('legus_ngc628c.txt')

    # Map the catalog's bands to the library's filter names
    bands = {'F275W': 'HST.WFC3_UVIS1.F275W',
             'F336W': 'HST.WFC3_UVIS1.F336W',
             'F435W': 'HST.ACS_WFC.F435W',
             'F555W': 'HST.ACS_WFC.F555W',
             'F814W': 'HST.ACS_WFC.F814W'}
    filters = list(bands.values())

    # Absolute magnitudes and errors, one row per cluster and one column
    # per band
    dist_mod = 29.98
    phot = np.column_stack([obs[b] for b in bands]) - dist_mod
    phot_err = np.column_stack([obs[b + '_err'] for b in bands])
    return filters, obs, phot, phot_err


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The prior

    By default, `cluster_slug` adopts as its prior the distribution from which
    the library's clusters were drawn. Here we instead choose our own: a
    cluster mass distribution $dN/dM \propto M^{-1.5}$, an age distribution
    $dN/dT \propto T^{-0.5}$, and a Gaussian distribution of [Fe/H], centered
    on 0 with a dispersion of 0.2 (and, since it does not depend on $A_V$, a
    flat distribution of $A_V$). We give the prior as a function, which
    `cluster_slug` calls with the mass, age, and $A_V$ (as astropy Quantities)
    and [Fe/H] (as a numpy array) of every cluster in the library. It must
    return the prior probability density in linear, not logarithmic, mass and
    age; `cluster_slug` divides it by the density from which the library was
    sampled to weight each library cluster.
    """)
    return


@app.cell
def _(np, u):
    def prior(mass, age, av, feh):
        """Prior density: dN/dM ~ M^-1.5, dN/dT ~ T^-0.5, Gaussian in [Fe/H]."""
        return (mass.to_value(u.Msun)**-1.5 * age.to_value(u.yr)**-0.5
                * np.exp(-feh**2 / (2 * 0.2**2)))

    return (prior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The library

    Next we load the default library, with our prior, and build the kernel
    density estimate for our five filters, using a bandwidth of 0.1 dex in
    the physical quantities and 0.1 mag in the photometry. The library holds
    30 million clusters, so this step takes a little while.
    """)
    return


@app.cell
def _(cluster_slug, filters, prior):
    # Load the default library, with our prior
    cs = cluster_slug(bw_phot=0.1, bw_phys=0.1, priors=prior)

    # Build the kernel density estimate for our filters
    cs.add_filterset(filters)
    return (cs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posteriors

    Now we condition the kernel density estimate on each cluster's
    photometry, with its errors; this returns one posterior PDF of the
    physical properties (log mass, log age, $A_V$, and [Fe/H]) per cluster,
    computed in parallel.
    """)
    return


@app.cell
def _(cs, phot, phot_err):
    # Posterior PDFs for every cluster
    posts = cs.conditional(phot, errors=phot_err)
    return (posts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nearest neighbors

    A posterior is only as good as the library's coverage of the observed
    photometry, so as a check we find each cluster's five nearest neighbors
    in the library. With `norm="errors"`, distances are in units of the
    photometric errors: the root-mean-square difference per band between the
    observed and library photometry, each band's difference divided by its
    error. A cluster whose fifth-nearest neighbor is within a distance of a
    few has several library clusters that match its photometry about as well
    as its errors allow; one whose nearest neighbors are much farther away is
    not well reproduced by the library, and its posterior should be treated
    with caution.
    """)
    return


@app.cell
def _(cs, phot, phot_err):
    # Distances to, and library rows of, each cluster's five nearest
    # neighbors in the library, in units of its photometric errors
    nn_dist, nn_rows = cs.neighbors(phot, 5, errors=phot_err, norm='errors')
    return (nn_dist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Marginal percentiles

    Finally, we marginalize each posterior down to one dimension, in log mass
    and in log age, and find percentiles of each marginal PDF. We tabulate
    these along with the distance to each cluster's fifth-nearest neighbor,
    `d5`.
    """)
    return


@app.cell
def _(Table, nn_dist, np, obs, posts):
    # Percentiles to compute
    pcts = [16, 25, 50, 75, 84]

    # For each cluster, the percentiles of its marginal PDFs of log mass
    # and log age
    quantities = {'logM': 'mass', 'logT': 'age'}
    results = Table()
    results['ID'] = obs['ID']
    for _label, _qty in quantities.items():
        _vals = np.zeros((len(posts), len(pcts)))
        for _i, _post in enumerate(posts):
            _others = [_n for _n in _post.names if _n != _qty]
            _vals[_i] = _post.marginalize(_others).percentile(0, pcts)
        for _j, _p in enumerate(pcts):
            results[f'{_label}_{_p}'] = _vals[:, _j]
            results[f'{_label}_{_p}'].format = '.2f'
    results['d5'] = nn_dist[:, 4]
    results['d5'].format = '.2f'
    results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The full posterior of one cluster

    To see the full posterior PDF of the first cluster, over all four
    physical quantities, we make a corner plot of it. For this we need the
    points and weights from which its posterior kernel density estimate is
    built. We could take these from its `KDE` in `posts`, but instead we use
    `conditional_sample`, which returns the same points, weights, and the
    rows of the library they come from, without building a `KDE` at all.
    This is convenient for, e.g., storing posteriors in a file, from which
    the `KDE` objects can be rebuilt later as needed. The `massTol` keyword
    specifies how much of the probability mass can be dropped when getting
    the sample points -- setting `massTol = 1e-3` means that the sample
    points returned must account for at least 99.9% of the probability mass.
    """)
    return


@app.cell
def _(cs, phot, phot_err):
    # Sample points, weights, and library rows of the first cluster's
    # posterior; the points' dimensions are the inference quantities, in
    # order (log mass, log age, A_V, [Fe/H]).
    sample_pts, sample_wgts, sample_rows = cs.conditional_sample(phot[0], errors=phot_err[0], massTol=1e-3)
    return sample_pts, sample_wgts


@app.cell
def _(npdfplot, plt, sample_pts, sample_wgts):
    # Corner plot of the first cluster's posterior; the outer contour in each panel encloses the 95% confidence region
    npdfplot(sample_pts, 
             weights=sample_wgts, 
             labels=[r'$\log M$ [M$_\odot$]', r'$\log T$ [yr]', r'$A_V$ [mag]', '[Fe/H]'],
             lolim=[2.5, 6.5, 0, -0.7],
             hilim=[4.5, 9, 1, 0.3],
             nbin=20,
             threshfrac=0.95,
             cbarmin=0,
             log=False,
             scat_skip=10,
             scat_alpha=0.25)
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
