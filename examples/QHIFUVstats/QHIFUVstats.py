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

    return (plt,)


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
    return


@app.cell
def _():
    # Print list of filters for which photometry is available

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
