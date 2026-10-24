"""
npdfplot.py

Implements npdfplot, which makes a corner plot of a set of points in
several dimensions: a histogram of each dimension along the diagonal,
and a 2D PDF of each pair of dimensions below it, shown as a density
map where the points are dense enough and as individual points where
they are not.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import Colormap
from matplotlib.image import AxesImage
from numpy.typing import ArrayLike
from scipy.stats import binned_statistic_2d

ZScaling = Literal["max", "count", "frac", "density", "normed"]

# Axis label for each choice of zscaling
_ZSCALING_LABELS: dict[str, str] = {
    "max": "Scaled PDF",
    "count": "N",
    "frac": "Fraction",
    "density": "Density",
    "normed": "PDF",
}


def _nbins(nbin: int | Sequence[int], i: int) -> int:
    """
    The number of bins in one dimension.

    Parameters
    ----------
    nbin : int or sequence of int
        The number of bins: the same in every dimension, or one per
        dimension.
    i : int
        The dimension.

    Returns
    -------
    int
        The number of bins in dimension i.
    """
    return nbin if isinstance(nbin, int) else int(nbin[i])


def _limits(x: np.ndarray, i: int, lolim: np.ndarray | None,
            hilim: np.ndarray | None) -> tuple[float, float]:
    """
    The plotting range in one dimension.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        The points' coordinates in dimension i.
    i : int
        The dimension.
    lolim, hilim : numpy.ndarray or None
        Lower and upper limits in every dimension; where None, the
        minimum and maximum of x.

    Returns
    -------
    tuple of float
        The lower and upper limits in dimension i.
    """
    lo = float(np.amin(x)) if lolim is None else float(lolim[i])
    hi = float(np.amax(x)) if hilim is None else float(hilim[i])
    return lo, hi


def npdfplot(pts: ArrayLike, fignum: int | None = None, lolim: ArrayLike | None = None,
             hilim: ArrayLike | None = None, nbin: int | Sequence[int] = 50,
             thresh: float | None = None, threshfrac: float = 0.99, log: bool = True,
             cbarmin: float | None = None, zmax: float | None = None,
             zscaling: ZScaling = "max", cmap: str | Colormap = "afmhot_r",
             histlim: ArrayLike | None = None, nohistlabel: bool = False,
             colorbar: bool = True, nticks: int | None = None,
             labels: Sequence[str] | None = None, scat_alpha: float = 1.0,
             scat_size: float = 5, scat_skip: int = 1, truths: ArrayLike | None = None,
             weights: ArrayLike | None = None) -> None:
    """
    Make a corner plot.

    Parameters
    ----------
    pts : array_like, shape (N, M)
        The points from which to make the plot: N points in M
        dimensions.
    fignum : int, optional
        Number of the figure in which to make the plot (default: a new
        figure).
    lolim : array_like, shape (M,), optional
        Lower limit to plot in each dimension (default: the minimum of
        the points).
    hilim : array_like, shape (M,), optional
        Upper limit to plot in each dimension (default: the maximum of
        the points).
    nbin : int or sequence of int, optional
        Number of histogram bins: the same in every dimension if an
        int, or one per dimension (shape (M,)) if a sequence (default
        50).
    thresh : float, optional
        Value of the 2D PDFs below which to show individual points
        rather than the density map (default: set by threshfrac).
    threshfrac : float, optional
        If thresh is None, the threshold is set so that this fraction
        of the points lie in regions above it, and so are shown by the
        density map; the rest are shown individually (default 0.99).
    log : bool, optional
        Whether to show the PDFs on a logarithmic (the default) or a
        linear scale.
    cbarmin : float, optional
        Minimum of the color scale of the 2D PDFs (default: the
        threshold).
    zmax : float, optional
        Maximum of the color scale of the 2D PDFs (default: 1 if
        zscaling is "max", otherwise the largest value of any 2D PDF).
    zscaling : {"max", "count", "frac", "density", "normed"}, optional
        How to scale the PDFs: "max" (the default) normalizes each to a
        maximum of unity; "count" shows the number of points in each
        bin; "frac" the fraction of the points in each bin; "density"
        the density of points in each bin; and "normed" the probability
        density in each bin.
    cmap : str or matplotlib.colors.Colormap, optional
        Color map for the 2D PDFs (default "afmhot_r").
    histlim : array_like, shape (2,) or (M, 2), optional
        Lower and upper limits of the histograms on the diagonal: the
        same for every histogram, or one pair per dimension (default:
        matplotlib's own).
    nohistlabel : bool, optional
        If True, omit the histograms' axis labels (default False).
    colorbar : bool, optional
        Whether to add a color bar for the 2D PDFs (default True).
    nticks : int, optional
        Number of tick marks on the color bar (default: matplotlib's
        own).
    labels : sequence of str, shape (M,), optional
        Axis label for each dimension (default: none).
    scat_alpha : float, optional
        Alpha of the individually shown points (default 1).
    scat_size : float, optional
        Marker size of the individually shown points (default 5).
    scat_skip : int, optional
        Show only every scat_skip-th of the individually shown points
        (default 1, all of them).
    truths : array_like, shape (M,), optional
        True values of the parameters, to mark on the plot (default:
        none).
    weights : array_like, shape (N,), optional
        Weight of each point (default: equal weights).

    Raises
    ------
    ValueError
        If zscaling is not one of the values above.
    """
    if zscaling not in _ZSCALING_LABELS:
        raise ValueError(f"bad value of zscaling: {zscaling!r}")
    points = np.asarray(pts, dtype=float)
    ndim = points.shape[-1]
    lo = None if lolim is None else np.asarray(lolim, dtype=float)
    hi = None if hilim is None else np.asarray(hilim, dtype=float)
    wgts = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=float)
    wsum = float(len(points)) if weights is None else float(np.sum(wgts))

    # Holders for the histograms, 2D PDFs, the bins each point falls in,
    # and which points to show individually, for each dimension or pair
    histarr: list[np.ndarray] = []
    zarr: list[np.ndarray] = []
    binidxarr: list[np.ndarray] = []
    scatteridxarr: list[np.ndarray] = []

    # First pass over dimension pairs: binning
    for i in range(ndim):

        # Get x limits and bins
        x = points[:, i]
        nxbin = _nbins(nbin, i)
        xlim = _limits(x, i, lo, hi)

        # Do 1D binning, then apply scaling and save
        hist, xe = np.histogram(x, bins=nxbin, range=xlim,
                                weights=None if weights is None else wgts)
        if zscaling == "max":
            histarr.append(hist / float(np.amax(hist)))
        elif zscaling == "count":
            histarr.append(hist)
        elif zscaling == "frac":
            histarr.append(hist / wsum)
        elif zscaling == "density":
            histarr.append(hist / (xe[1] - xe[0]))
        else:  # normed
            histarr.append(hist / (wsum * (xe[1] - xe[0])))

        # Loop in j direction
        for j in range(i + 1, ndim):

            # Get y limits and bins
            y = points[:, j]
            nybin = _nbins(nbin, j)
            ylim = _limits(y, j, lo, hi)

            # Make 2D histogram
            result: Any = binned_statistic_2d(x, y, wgts, statistic="sum",
                                              bins=[nxbin, nybin], range=[xlim, ylim],
                                              expand_binnumbers=True)
            count = np.asarray(result[0])
            xe2 = np.asarray(result[1])
            ye2 = np.asarray(result[2])
            binidx = np.asarray(result[3])

            # Set z
            if zscaling == "max":
                z = count / np.amax(count)
            elif zscaling == "count":
                z = count
            elif zscaling == "frac":
                z = count / len(points)
            elif zscaling == "density":
                z = count / ((xe2[1] - xe2[0]) * (ye2[1] - ye2[0]))
            else:  # normed
                z = count / (len(points) * (xe2[1] - xe2[0]) * (ye2[1] - ye2[0]))

            # Save
            zarr.append(z)
            binidxarr.append(binidx)

    # Second pass: set maxima and minima for each dimension pair, then
    # homogenize them
    zmaxarr: list[float] = []
    zminarr: list[float] = []
    for z in zarr:
        if zmax is not None:
            zmaxarr.append(zmax)
        elif zscaling == "max":
            zmaxarr.append(1.0)
        else:
            zmaxarr.append(float(np.amax(z)))
        if thresh is not None:
            zminarr.append(thresh)
        else:
            zsort = np.sort(z, axis=None)
            csum = np.cumsum(zsort)
            csum = csum / csum[-1]
            zminarr.append(float(zsort[np.argmax(csum > 1.0 - threshfrac)]))
    zhi = float(np.amax(zmaxarr)) if zmaxarr else 1.0
    zlo = float(np.amin(zminarr)) if zminarr else 0.0
    if log:
        zhi = float(np.log10(zhi))
        zlo = float(np.log10(zlo))

    # Third pass: figure out which points to show individually -- those
    # within the binned region but in bins below the minimum
    for z, binidx in zip(zarr, binidxarr, strict=True):
        flag = np.logical_and.reduce((binidx[0, :] > 0,
                                      binidx[1, :] > 0,
                                      binidx[0, :] <= z.shape[0],
                                      binidx[1, :] <= z.shape[1]))
        scatteridx = np.zeros(len(points), dtype=bool)
        zcut = 10.0**zlo if log else zlo
        scatteridx[flag] = z[binidx[0, flag] - 1, binidx[1, flag] - 1] < zcut
        scatteridxarr.append(scatteridx)

    # Set up plot
    plt.figure(fignum)
    plt.clf()
    gs = gridspec.GridSpec(ndim, ndim)

    # Fourth pass: loop over dimension pairs, this time to make the plot
    img: AxesImage | None = None
    ctr = 0
    for i in range(ndim):

        # Make grid in x direction
        x = points[:, i]
        nxbin = _nbins(nbin, i)
        xlim = _limits(x, i, lo, hi)
        xgrd = np.linspace(xlim[0], xlim[1], nxbin)

        # Make 1D histogram on the diagonal
        ax = plt.subplot(gs[i, i])
        ax.bar(xgrd, histarr[i], xgrd[1] - xgrd[0], align="center",
               facecolor="C0", edgecolor="black")
        ax.set_xlim(xlim)
        if log:
            ax.set_yscale("log")

        # Set histogram limits
        if histlim is not None:
            hl = np.asarray(histlim, dtype=float)
            row = hl[i, :] if hl.ndim == 2 else hl
            ax.set_ylim(float(row[0]), float(row[1]))

        # Add truths line
        if truths is not None:
            tr = np.asarray(truths, dtype=float)
            ax.plot([tr[i], tr[i]], ax.get_ylim(), color="C0", lw=1)

        # Move axis and add label
        ax.yaxis.tick_right()
        if not nohistlabel:
            ax.set_ylabel(_ZSCALING_LABELS[zscaling])
        ax.yaxis.set_label_position("right")

        # Set x labels
        if i == ndim - 1 and labels is not None:
            ax.set_xlabel(labels[i])
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)
        else:
            ax.set_xticklabels([])

        # Loop in j direction
        for j in range(i + 1, ndim):

            # Grab axes for this pair
            ax = plt.subplot(gs[j, i])

            # Define y grid and 2D grid
            y = points[:, j]
            nybin = _nbins(nbin, j)
            ylim = _limits(y, j, lo, hi)
            ygrd = np.linspace(ylim[0], ylim[1], nybin)
            xx, yy = np.meshgrid(xgrd, ygrd)

            # If using logarithmic scaling, apply it
            z = zarr[ctr]
            if log:
                z[z == 0] = 1.0e-6 * np.amin(z[z > 0])
                z = np.log10(z)

            # Plot contour at threshold
            ax.contour(xx, yy, np.transpose(z), levels=[zlo], colors="k",
                       linestyles="-")

            # Plot individual points outside contour
            ax.scatter(x[scatteridxarr[ctr]][::scat_skip],
                       y[scatteridxarr[ctr]][::scat_skip],
                       color="k", s=scat_size, alpha=scat_alpha, edgecolor="none")

            # Plot density map
            vmin = zlo if cbarmin is None else cbarmin
            img = ax.imshow(np.transpose(z), origin="lower", aspect="auto",
                            vmin=vmin, vmax=zhi, cmap=cmap,
                            extent=(xlim[0], xlim[1], ylim[0], ylim[1]))

            # Add truths
            if truths is not None:
                tr = np.asarray(truths, dtype=float)
                ax.plot([tr[i], tr[i]], ax.get_ylim(), color="C0", lw=1)
                ax.plot(ax.get_xlim(), [tr[j], tr[j]], color="C0", lw=1)
                ax.scatter([tr[i]], [tr[j]], marker="o", color="C0", s=40)

            # Set plot range
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # Adjust labels
            if i != 0:
                ax.set_yticklabels([])
            if j != ndim - 1:
                ax.set_xticklabels([])
            if labels is not None:
                if i == 0:
                    ax.set_ylabel(labels[j])
                if j == ndim - 1:
                    ax.set_xlabel(labels[i])
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)

            # Increment counter
            ctr = ctr + 1

    # Add final colorbar (only if there is a 2D PDF for it to describe)
    if colorbar and img is not None:
        label = _ZSCALING_LABELS[zscaling]
        if zscaling in ("frac", "density"):
            label = label.lower()
        if log:
            label = "log " + label
        ax = plt.subplot(gs[ndim // 2 - 1, ndim // 2 + 1]) if ndim > 2 else plt.subplot(gs[0, 1])
        cbar = plt.colorbar(img, ax=ax, orientation="vertical", fraction=0.8, aspect=10,
                            label=label)
        if nticks is not None:
            cbar.set_ticks(list(np.linspace(zlo, zhi, nticks)))
        ax.remove()
