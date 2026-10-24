# cluster_slug libraries

This directory holds libraries of slug cluster simulations for use with
`cluster_slug`, slug's tool for Bayesian inference of star cluster
properties from photometry. `default.h5` is the default library.

## Downloading the default library

`default.h5` is about 33 GB, so it is **not** downloaded automatically.
The data download script skips this directory unless you ask for it
explicitly with `--subdir cluster_slug`:

- From a checkout of the repository (installs into this directory):

  ```
  python3 data/tools/download_data.py --subdir cluster_slug
  ```

- From a pip-installed slugpy (installs into slug's per-user data
  directory, or into `$SLUG_DIR/data` if `SLUG_DIR` is set):

  ```
  python -m slugpy.download_data --subdir cluster_slug
  ```

Running the download script without `--subdir` fetches all the other
data (tracks, spectra, filters, etc.) and skips this directory.

## Contents of the default library

`default.h5` holds 30,000,000 simulated clusters. Each has one output
time, with no spectra saved, only photometry. The main settings are
listed below. The file itself holds the full detail, including the
input deck and the PDFs from which each property was drawn. Once
loaded, these are available from `cluster_slug`, e.g. through its
`pdfs` property.

| Setting | Value |
|---|---|
| IMF | Chabrier (2005): lognormal below 1 Msun (centered on 0.2 Msun, dispersion 0.55 dex), power law of slope -2.35 from 1 to 120 Msun; 0.08 - 120 Msun overall |
| Stellar tracks | MIST, rotating at v/v_crit = 0.4, [alpha/Fe] = 0 |
| Metallicity | [Fe/H] drawn uniformly from -1.0 to +0.3 |
| Cluster mass (CMF) | dN/dM ∝ M^-1 from 10^2 to 10^5 Msun, and ∝ M^-2 from 10^5 to 10^7 Msun |
| Cluster age | dN/dt ∝ t^-1 (uniform in log t) from 10^5 yr to 1.5 × 10^10 yr |
| Extinction | A_V drawn uniformly from 0 to 3 mag, Calzetti starburst extinction curve |
| Nebular emission | Included. The nebular emission's A_V is the stellar A_V times a factor drawn from a normal distribution of mean 2.1 and dispersion 0.5, truncated to the range 1 - 100. |
| Spectral synthesis | Default spectral model, 1024-point wavelength grid from 200 Å to 20 μm |
| Photometric system | Vega magnitudes |

Photometry is stored in four variants: stellar only, with extinction,
with nebular emission, and with both nebular emission and extinction.
`cluster_slug` uses the nebular, extincted variant by default.

The extincted photometry is unavailable (NaN) for JWST NIRCam F200W and
redder bands, and for the ionizing-photon filters, because the Calzetti
starburst curve does not extend to those wavelengths. For these bands,
`cluster_slug` falls back to the unextincted photometry.

### Filters

- **HST WFC3/UVIS:** F225W, F275W, F336W, F438W, F555W, F814W, F547M,
  F657N, F689M (`HST.WFC3_UVIS1.*`)
- **HST ACS/WFC:** F435W, F555W, F606W, F814W, F658N (`HST.ACS_WFC.*`)
- **JWST NIRCam:** F115W, F150W, F187N, F200W, F277W, F300M, F335M,
  F405N, F444W (`JWST.NIRCam.*`)
- **JWST MIRI:** F560W, F770W (`JWST.MIRI.*`)
- **Ionizing photon rates:** `Q(HI)`, `Q(HeI)`, `Q(OII)` (photons/s
  above the H, He, and O^+ ionization thresholds)
- **Bolometric luminosity:** `Lbol`
