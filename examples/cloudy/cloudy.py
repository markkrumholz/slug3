import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # Import required libraries
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    from astropy.constants import h, c

    return c, h, np, plt, u


@app.cell
def _():
    # Add slugpy to path, and import the simulation runner
    import sys
    import os
    sys.path = [os.path.join('..', '..')] + sys.path
    from slugpy import read

    return os, read


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
        sim_output = read('cloudy_example.h5')
    except FileNotFoundError:
        sim_output = run_sim('cloudy.toml')
    return (sim_output,)


@app.cell
def _(os, sim_output, u):
    # Check if the outputs already have cloudy run data; if not, run cloudy on
    # the first two clusters
    if sim_output.cluster_cloudy is None:
        # Path to cloudy executable on this system; replace with your own path
        # on your local system
        path_to_cloudy = os.path.join(os.getenv('CLOUDY_DIR'), "cloudy.exe")
        # Run cloudy on clusters 0 and 1
        sim_output.run_cloudy('cluster', cloudy_path=path_to_cloudy,
                              U=10.**-2.5, nII=100*u.cm**-3, uid=[0,1])
    return


@app.cell
def _(np, sim_output):
    # Grab total ionizing luminosities of all clusters, sorted by uid
    idx_phot = np.argsort(sim_output.cluster_phot['uid'])
    QHI = sim_output.cluster_phot['Q(HI)'][idx_phot]
    return QHI, idx_phot


@app.cell
def _(QHI, c, h, idx_phot, np, sim_output, u):
    # Print the luminosity per ionizing photon for the 10 brightest lines for clusters 0 and 1
    idx_cloudy = [np.where(sim_output.cluster_cloudy['uid']==0)[0], np.where(sim_output.cluster_cloudy['uid']==1)[0]]
    idx_line = np.argsort(np.squeeze(sim_output.cluster_cloudy['line_lum'][idx_cloudy[0]]))[::-1]
    print("{:4s}   {:15s}   {:15s}   {:18s}   {:15s}".format(
        'ID', 'Wavelength [Å]', 'Energy [eV]', '[L/Q]_1 [erg/ph]', '[L/Q]_2 [erg/ph]'))
    for i in range(10):
        label_str = sim_output.cluster_cloudy['line_label'][idx_line[i]].decode()
        wl_str = str(sim_output.cluster_cloudy['line_wl'][idx_line[i]].value)
        en = (h*c/sim_output.cluster_cloudy['line_wl'][idx_line[i]]).to(u.eV).value
        lum0 = (sim_output.cluster_cloudy['line_lum'][idx_cloudy[0],idx_line[i]][0] / QHI[idx_phot[0]]).value
        lum1 = (sim_output.cluster_cloudy['line_lum'][idx_cloudy[1],idx_line[i]][0] / QHI[idx_phot[0]]).value
        print(f"{label_str:4s}   {wl_str:15s}   {en:15.12f}   {lum0:18.12e}   {lum1:18.12e}")
    return (idx_cloudy,)


@app.cell
def _(QHI, c, h, idx_cloudy, np, plt, sim_output, u):
    # Plot the intrinsic stellar spectrum and the transmitted + emitted spectrum
    # for the two clusters, normalizing by Q(HI) to put them on the same scale

    # Grab indices
    idx_spec = [np.where(sim_output.cluster_spectra['uid']==0)[0], np.where(sim_output.cluster_spectra['uid']==1)[0]]

    # Grab wavelength grid
    wl_sp = sim_output.cluster_spectra['wl']   # slug spectral grid
    wl_cl = sim_output.cluster_cloudy['wl']   # cloudy spectral grid

    # Make plots
    for j in range(2):

        # Open plot window
        plt.subplot(2,1,j+1)

        # Plot spectra
        L_lam_stellar = np.squeeze(sim_output.cluster_spectra['spec'][idx_spec[j],:])
        L_lam_neb = np.squeeze(sim_output.cluster_cloudy['spec_trans_emit'][idx_cloudy[j],:])
        plt.loglog(wl_sp, wl_sp * L_lam_stellar / QHI[j], 
                   color='C0'.format(j), ls='--',
                   label='Intrinsic stellar') 
        plt.loglog(wl_cl, wl_cl * L_lam_neb / QHI[j],
                   color='C0'.format(j),
                   label='Transmitted stellar + nebular') 

        # Annotate lines
    
    
        # Adjust range
        wl_lim = np.array([100,1e8])*u.Angstrom
        plt.xlim(wl_lim.value)
        plt.ylim([3e-14,3e-8])

        # Adjust axis labels
        if j == 0:
            plt.gca().set_xticklabels('')
        else:
            plt.xlabel(r'$\lambda$ [$\AA$]')
        plt.ylabel(r'$\lambda L_\lambda/Q(\mathrm{HI})$ [erg ph$^{-1}$]')

        # Add legend
        plt.legend(loc='upper right')

        # Add upper axis label
        plt.twiny()
        plt.xscale('log')
        plt.xlim((h*c/wl_lim).to(u.eV).value)
        if j == 0:
            plt.xlabel(r'$E_\gamma$ [eV]')
        else:
            plt.gca().set_xticklabels('')

    # Show plot
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
