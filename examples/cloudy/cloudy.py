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

    return np, plt, u


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
    # Check if the outputs already have cloudy run data; if not, run cloudy
    if sim_output.cluster_cloudy is None:
        # Path to cloudy executable on this system; replace with your own path
        # on your local system
        path_to_cloudy = os.path.join(os.getenv('CLOUDY_DIR'), "cloudy.exe")
        # Run cloudy
        sim_output.run_cloudy('cluster', cloudy_path=path_to_cloudy,
                              U=10.**-2.5, nII=100*u.cm**-3, uid=0)
    return (path_to_cloudy,)


@app.cell
def _(path_to_cloudy, sim_output, u):
    sim_output.run_cloudy('cluster', cloudy_path=path_to_cloudy,
                              U=10.**-2.5, nII=100*u.cm**-3, uid=1)
    return


@app.cell
def _(sim_output):
    sim_output.cluster_cloudy.keys()
    return


@app.cell
def _(plt, sim_output):
    plt.loglog(sim_output.cluster_spectra['wl'], sim_output.cluster_spectra['wl']*sim_output.cluster_spectra['spec'][0,:])
    plt.loglog(sim_output.cluster_cloudy['wl'], sim_output.cluster_cloudy['wl']*sim_output.cluster_cloudy['spec_trans_emit'][0,:])
    #plt.loglog(sim_output.cluster_cloudy['wl'], sim_output.cluster_cloudy['wl']*sim_output.cluster_cloudy['spec_inc'][0,:])
    plt.xlim([100,1e8])
    plt.ylim([1e37,1e44])
    plt.show()
    return


@app.cell
def _(np, sim_output):
    idx = np.argsort(sim_output.cluster_cloudy['line_lum'])[::-1]
    for i in range(10):
        print(sim_output.cluster_cloudy['line_label'][0])
    return


@app.cell
def _(sim_output):
    sim_output.cluster_cloudy['spec_trans_emit'].shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
