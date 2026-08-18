import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


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
    from scipy.stats import binned_statistic

    return binned_statistic, np, plt, u


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
        sim_output = read('QHIFUVstats.h5')
    except FileNotFoundError:
        sim_output = run_sim('QHIFUVstats.toml')
    return (sim_output,)


@app.cell
def _(np, sim_output):
    # Grab the SFR (= target_mass / time) and sort by trial number
    sfr_trial = sim_output.galaxy['trial']
    sfr = sim_output.galaxy['target_mass'] / sim_output.galaxy['time']
    sfr = sfr[np.argsort(sfr_trial)]
    return (sfr,)


@app.cell
def _(np, sim_output):
    # Grab FUV and ionizing luminosities and sort by trial number
    phot_trial = sim_output.galaxy_phot['trial']
    l_fuv = sim_output.galaxy_phot['GALEX.GALEX.FUV'][np.argsort(phot_trial)]
    q_hi = sim_output.galaxy_phot['Q(HI)'][np.argsort(phot_trial)]
    return l_fuv, q_hi


@app.cell
def _(binned_statistic, l_fuv, np, q_hi, sfr, u):
    # Compute l_fuv/sfr and q_hi/sfr in bins of log(sfr)
    l_fuv_norm = (l_fuv / sfr).to(u.erg/u.s/u.Angstrom / (u.Msun/u.yr))
    q_hi_norm = (q_hi / sfr).to(u.ph/u.s / (u.Msun/u.yr))

    # Set up binning in log(sfr)
    log_sfr_range = [-6, -2]
    nbin = 16
    log_sfr = np.log10(sfr / (u.Msun/u.yr))

    # Compute median, mean and 25th-75th percentile range for FUV
    l_fuv_norm_med, log_sfr_edges, *_ = binned_statistic(
        log_sfr, l_fuv_norm.value, statistic='median', 
        range=log_sfr_range, bins=nbin)
    l_fuv_norm_mean, *_ = binned_statistic(log_sfr, l_fuv_norm.value, 
                                           statistic='mean', 
                                           range=log_sfr_range, bins=nbin)
    l_fuv_norm_25, *_ = binned_statistic(log_sfr, l_fuv_norm.value,
                                        statistic=lambda x: np.percentile(x, 25),
                                        range=log_sfr_range, bins=nbin)
    l_fuv_norm_75, *_ = binned_statistic(log_sfr, l_fuv_norm.value,
                                        statistic=lambda x: np.percentile(x, 75),
                                        range=log_sfr_range, bins=nbin)

    # Repeat for qhi
    q_hi_norm_med, log_sfr_edges, *_ = binned_statistic(
        log_sfr, q_hi_norm.value, statistic='median', 
        range=log_sfr_range, bins=nbin)
    q_hi_norm_mean, *_ = binned_statistic(log_sfr, q_hi_norm.value, 
                                          statistic='mean', 
                                          range=log_sfr_range, bins=nbin)
    q_hi_norm_25, *_ = binned_statistic(log_sfr, q_hi_norm.value,
                                        statistic=lambda x: np.percentile(x, 25),
                                        range=log_sfr_range, bins=nbin)
    q_hi_norm_75, *_ = binned_statistic(log_sfr, q_hi_norm.value,
                                        statistic=lambda x: np.percentile(x, 75),
                                        range=log_sfr_range, bins=nbin)

    # Compute overall means
    l_fuv_norm_mean_tot = np.mean(l_fuv_norm)
    q_hi_norm_mean_tot = np.mean(q_hi_norm)
    return (
        l_fuv_norm_25,
        l_fuv_norm_75,
        l_fuv_norm_mean,
        l_fuv_norm_mean_tot,
        l_fuv_norm_med,
        log_sfr_edges,
        log_sfr_range,
        q_hi_norm_25,
        q_hi_norm_75,
        q_hi_norm_mean,
        q_hi_norm_mean_tot,
        q_hi_norm_med,
    )


@app.cell
def _(
    l_fuv_norm_25,
    l_fuv_norm_75,
    l_fuv_norm_mean,
    l_fuv_norm_mean_tot,
    l_fuv_norm_med,
    log_sfr_edges,
    log_sfr_range,
    np,
    plt,
    q_hi_norm_25,
    q_hi_norm_75,
    q_hi_norm_mean,
    q_hi_norm_mean_tot,
    q_hi_norm_med,
):
    sfr_bin_ctr = 10.**((log_sfr_edges[1:] + log_sfr_edges[:-1]) / 2)

    # FUV
    plt.plot(sfr_bin_ctr, l_fuv_norm_mean/l_fuv_norm_mean_tot, color='C0', ls='--', label='mean(FUV)')
    plt.plot(sfr_bin_ctr, l_fuv_norm_med/l_fuv_norm_mean_tot, color='C0', label='med(FUV)')
    plt.fill_between(sfr_bin_ctr, 
                     l_fuv_norm_25/l_fuv_norm_mean_tot.value, 
                     l_fuv_norm_75/l_fuv_norm_mean_tot.value,
                     alpha=0.3, color='C0', lw=0, label='IQR(FUV)')

    # Q(HI)
    plt.plot(sfr_bin_ctr, q_hi_norm_mean/q_hi_norm_mean_tot, color='C1', ls='--', label='mean(Q(HI))')
    plt.plot(sfr_bin_ctr, q_hi_norm_med/q_hi_norm_mean_tot, color='C1', label='med(Q(HI))')
    plt.fill_between(sfr_bin_ctr, 
                     q_hi_norm_25/q_hi_norm_mean_tot.value, 
                     q_hi_norm_75/q_hi_norm_mean_tot.value,
                     alpha=0.3, color='C1', lw=0, label='IQR(Q(HI))')

    # Adjust scales, add axis labels
    plt.xlim(10.**np.array(log_sfr_range))
    plt.ylim([1e-2,2])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('SFR [M$_\odot$ yr$^{-1}$]')
    plt.ylabel(r'$L/\mathrm{SFR} / \left\langle L/\mathrm{SFR}\right\rangle$')

    # Add legend
    plt.legend(loc='lower right', ncol=2)

    # Show
    plt.show()
    return


if __name__ == "__main__":
    app.run()
