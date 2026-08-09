import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # Add slugpy to path, and import the simulation runner
    import sys
    import os.path as osp
    sys.path = [osp.join('..', '..')] + sys.path
    from slugpy import run_sim

    return (run_sim,)


@app.cell
def _(run_sim):
    # Run the simulation
    sim_output = run_sim('QHIstats.toml')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
