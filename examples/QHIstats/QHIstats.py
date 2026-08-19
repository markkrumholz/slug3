import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Copyright (c) 2026 Mark Krumholz. All rights reserved.
    """)
    return


@app.cell
def _():
    # Import marimo for markdown rendering
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Add slugpy to path, and import the simulation runner
    import sys
    import os.path as osp
    sys.path = [osp.join('..', '..')] + sys.path
    from slugpy import run_sim, read

    return read, run_sim


@app.cell
def _():
    # Import other tools
    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    from astropy.constants import Ryd
    from scipy.stats import binned_statistic

    return Ryd, binned_statistic, np, plt, u


@app.cell
def _(plt):
    # Configure matplotlib to plot using pretty fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=12)
    return


@app.cell
def _(read, run_sim):
    # Check if the outputs already exist; if they do, load them, and
    # if not, run the simulation to generate them
    try:
        sim_output = read('QHIstats.h5')
    except FileNotFoundError:
        sim_output = run_sim('QHIstats.toml')
    return (sim_output,)


@app.cell
def _(np, sim_output):
    # Grab cluster masses and the id numbers they go with; sort by
    # id number, since id's may be out of order
    mass = sim_output.clusters['birth_mass']
    mass_id = sim_output.clusters['uid']
    mass = mass[np.argsort(mass_id)]
    return (mass,)


@app.cell
def _(np, sim_output):
    # Grab ionizing luminosities and the id numbers they go with; sort by id
    qHI = sim_output.cluster_phot['Q(HI)']
    qHI_id = sim_output.cluster_phot['uid']
    qHI = qHI[np.argsort(qHI_id)]
    return (qHI,)


@app.cell
def _(np, sim_output):
    # Grab specific luminosities, and sort by id number
    wl = sim_output.cluster_spectra['wl']
    lspec = sim_output.cluster_spectra['spec'] * wl
    lspec_id = sim_output.cluster_spectra['uid']
    lspec = lspec[np.argsort(lspec_id)]
    return lspec, wl


@app.cell
def _(binned_statistic, mass, np, plt, qHI, u):
    # Compute ionizing luminosity normalized by mass
    qHI_norm = qHI / mass

    # Compute mean and percentiles of qHI_norm in mass bins
    logm = np.log10(mass/u.Msun)
    nbin = 5
    qHI_norm_med, logm_edges, *_ = binned_statistic(logm, qHI_norm, statistic='median', bins=nbin)
    qHI_norm_25, *_ = binned_statistic(logm, qHI_norm, statistic=lambda x: np.percentile(x, 25), bins=nbin)
    qHI_norm_75, *_ = binned_statistic(logm, qHI_norm, statistic=lambda x: np.percentile(x, 75), bins=nbin)
    qHI_norm_mean, *_ = binned_statistic(logm, qHI_norm, statistic='mean', bins=nbin)

    # Plot median, mean, and interquartile range
    norm_fac = 1e46
    logm_cen = (logm_edges[:-1] + logm_edges[1:])/2
    plt.plot(logm_cen, qHI_norm_med/norm_fac, label='Median')
    plt.fill_between(logm_cen, qHI_norm_75/norm_fac, qHI_norm_25/norm_fac, 
                     color='C0', alpha=0.3, lw=0,
                     label='IQR')
    plt.plot(logm_cen, qHI_norm_mean/norm_fac, label='Mean')

    # Adjust axes
    plt.xlim([logm_cen[0], logm_cen[-1]])
    plt.ylim([0, 8])

    # Add axis labels
    plt.xlabel(r'$\log(M/\mathrm{M}_\odot)$')
    plt.ylabel(r'$Q(\mathrm{HI})/M$ [$10^{46}$ ph s$^{-1}$ M$_\odot^{-1}$]')

    # Add legend
    plt.legend()

    # Show plot
    plt.show()
    return


@app.cell
def _(Ryd, lspec, mass, np, plt, u, wl):
    # Compute the mass-normalized specific luminosity
    lspec_norm = (lspec / mass[:, np.newaxis]).to(u.Lsun/u.Msun)

    # Compute the median mass-normalized specific luminosity and the interquartile range around it
    lspec_norm_pct = np.percentile(lspec_norm, [25, 50, 75], axis=0)

    # Plot the median and range
    plt.loglog(wl, lspec_norm_pct[1,:], color='C0', label='Median')
    plt.fill_between(wl.value,   # .value required because astropy units don't play nicely with fill_between
                     lspec_norm_pct[0,:].value, 
                     lspec_norm_pct[2,:].value,
                     alpha=0.3, color='C0', lw=0, label='IQR')

    # Add shading showing Lyman edge
    lam_LyE = (1/Ryd).to(u.Angstrom)
    plt.plot(lam_LyE*np.ones(2), [1, 3e3], 'k--')
    plt.fill_between([wl[0].value, lam_LyE.value], [3e3, 3e3], 
                     color='k', alpha=0.1, lw=0)

    # Add labels
    plt.xlabel(r'$\lambda$ [$\AA$]')
    plt.ylabel(r'$\lambda L_\lambda/M$ [L$_\odot$ M$_\odot^{-1}$]')
    plt.text(lam_LyE.value, 10, 
             r'$\lambda = {:5.1f}$ $\AA$'.format(lam_LyE.value), 
             rotation=-90)

    # Adjust plot range
    plt.xlim([wl[0].value, wl[-1].value])
    plt.ylim([1, 3e3])

    # Add legend
    plt.legend()

    # Show image
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
