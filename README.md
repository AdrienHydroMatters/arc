# Bayesian Rating Curve Estimation from Satellite Altimetry and In-Situ Discharge

## Overview

This repository provides a standalone Python tool for estimating **power-law rating curves** from paired Water Surface Elevation (WSE) and river discharge (Q) time series. The approach combines Bayesian Markov Chain Monte Carlo (MCMC) inference with a physically constrained zero-flow datum (*z*₀), enabling uncertainty-aware parameter estimation suitable for large-sample hydrological studies.

The method is designed to work with satellite altimetry WSE data (e.g. Jason-3, Sentinel-6) and global or national discharge archives (e.g. GRDC, HydroWeb). All outputs are structured CSV files — no plots are generated.

---

## Rating Curve Model

The power-law rating curve takes the form:

$$Q = a \cdot (h - z_0)^{b}$$

where *Q* is river discharge (m³ s⁻¹), *h* is water surface elevation (m), *z*₀ is the effective zero-flow datum (m), and *a*, *b* are shape parameters. Parameters are estimated as posterior medians from a Bayesian MCMC sampler with weakly informative priors, and uncertainty is reported as posterior standard deviations.

---

## Repository Structure

```
.
├── compute_arc.py           # Main script — Bayesian RC estimation
├── environment.yml          # conda environment definition (arc)
├── setup_env.sh             # one-command environment setup helper
├── README.md                # This file
├── data/
│   ├── WSE/                 # Water Surface Elevation time series (one file per station)
│   │   └── WSE_{BASIN}_{STATION}[_suffix].txt
│   ├── Q/                   # Discharge time series (one file per station)
│   │   └── Q_{BASIN}_{STATION}[_suffix].txt
│   └── Zmin.csv             # Optional per-station z0 constraint table
└── results/                 # Output folder (auto-created)
    ├── rating_curve_summary.csv
    └── validation.csv
```

---

## Input Data Format

Both WSE and Q files are **semicolon-delimited** with a mandatory header. Each file contains the complete time series for one gauging or virtual station.

### File naming convention

| Type | Pattern | Example |
|---|---|---|
| WSE | `WSE_{BASIN}_{STATION}[_suffix].txt` | `WSE_NIGER_MALANVILLE_JASON3.txt` |
| Discharge | `Q_{BASIN}_{STATION}[_suffix].txt` | `Q_NIGER_MALANVILLE.txt` |

WSE and Q files are paired automatically by **bidirectional longest-prefix match** on the `{BASIN}_{STATION}` token. Either file may carry an extra suffix the other does not.

**Examples of valid pairs**

| WSE file | Matched Q file |
|---|---|
| `WSE_ADOUR_ADOUR-KM0115-EXP.txt` | `Q_ADOUR_ADOUR-KM0115-EXP_rivid-23024083.txt` |
| `WSE_NIGER_MALANVILLE_JASON3.txt` | `Q_NIGER_MALANVILLE.txt` |
| `WSE_DANUBE_MOHACS.txt` | `Q_DANUBE_MOHACS.txt` |

### Column description

| Column | Unit | Description |
|---|---|---|
| `station` | — | Station identifier |
| `lon` | decimal degrees | Longitude (WGS84) |
| `lat` | decimal degrees | Latitude (WGS84) |
| `date` | — | Timestamp `YYYY-MM-DD HH:MM:SS` |
| `value` | m or m³ s⁻¹ | Observed WSE or discharge |
| `uncertainty` | same as value | Measurement uncertainty (1σ); `nan` if unavailable |
| `source` | — | Data provider (e.g. `grdc`, `Jason3`) |

### Zmin.csv — per-station z0 constraint

An optional CSV file can be used to supply a pre-defined *z*₀ upper constraint for each station, for instance derived from bathymetric surveys or a prior rating curve. It must be semicolon-delimited with the following columns:

```
station;lon;lat;zmin
NIGER_MALANVILLE;2.43;11.87;176.32
DANUBE_MOHACS;18.69;45.99;74.10
```

The `station` key must match `{BASIN}_{STATION}` (case-insensitive). If a station is absent from the file, `min(WSE)` is used as fallback.

> **Note on temporal coverage.** WSE and Q series do not need to span the same period. The script automatically finds the common overlap window (within ±24 h per pair) for calibration, and uses the full Q archive for climatology.

---

## Environment Setup

A ready-to-use conda environment named **`arc`** is provided.

**Requirements:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda ≥ 23.x

```bash
# Create the environment
bash setup_env.sh        # or: conda env create --name arc --file environment.yml

# Activate
conda activate arc

# Update an existing environment
bash setup_env.sh --update
```

Key dependencies: Python 3.11, NumPy ≥ 1.24, pandas ≥ 2.0, SciPy ≥ 1.11, PyMC ≥ 5.0, ArviZ ≥ 0.17, geopy ≥ 2.4.

---

## Usage

```bash
python compute_arc.py  <WSE_folder>  <Q_folder>  [options]
```

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `WSE_folder` | required | Folder containing `WSE*.txt` files |
| `Q_folder` | required | Folder containing `Q*.txt` files |
| `-z0 / --force_z0` | `None` | Path to a semicolon CSV (`station;lon;lat;zmin`) with per-station *z*₀ constraints. If omitted, `min(WSE)` is used. |
| `-o / --outpath` | `WSE_folder/rating_curve/` | Output directory |
| `-b / --basin` | `None` | Restrict to files matching a basin token (e.g. `NIGER`) |
| `-m / --min_points` | `12` | Minimum overlap pairs before falling back to the quantile approach |

### Examples

```bash
# Basic usage — z0 inferred from data
python compute_arc.py ./data/WSE ./data/Q

# Provide per-station z0 constraints from Zmin.csv
python compute_arc.py ./data/WSE ./data/Q -z0 ./data/Zmin.csv

# Process only Niger basin stations, save to custom folder
python compute_arc.py ./data/WSE ./data/Q -b NIGER -o ./results/NIGER

# Require at least 20 calibration points
python compute_arc.py ./data/WSE ./data/Q -m 20
```

---

## Outputs

All outputs are written to `<outpath>/`.

### `rating_curve_summary.csv` — one row per station

| Column | Description |
|---|---|
| `basin`, `station` | Station identifiers |
| `lon`, `lat` | Coordinates (WGS84) |
| `a`, `b`, `z0` | RC parameters — posterior medians |
| `a_sd`, `b_sd`, `z0_sd` | Posterior standard deviations |
| `zmin` | Minimum WSE of the calibration data (m) |
| `NSE`, `KGE`, `PBIAS`, `NRMSE`, `R2` | Performance metrics on climate monthly means |
| `approach` | Calibration strategy: `overlap` or `quantile` |
| `nb_cal_points` | Number of pairs used for fitting |
| `nb_clim_months` | Number of climate months used for summary statistics |

### `validation.csv` — one row per station (when available)

Statistics are computed on **year-monthly means** of *Q*RC vs. *Q*insitu over the common temporal overlap.

| Column | Description |
|---|---|
| `basin`, `station`, `lon`, `lat` | Station identifiers and coordinates |
| `NSE`, `KGE`, `PBIAS`, `NRMSE`, `R2` | Performance metrics |
| `val_start`, `val_end` | First and last `YYYY-MM` of the validation period |
| `nb_months` | Number of year-months compared |

---

## Methods

**Date matching.** WSE and Q observations are paired if their timestamps are within ±24 hours of each other (closest neighbour).

**Outlier filtering.** A local monotonicity filter in WSE-sorted order flags observations whose discharge falls more than 20 % below the median discharge of their 5 nearest WSE-space neighbours.

**z₀ constraint.** The zero-flow datum is physically constrained to remain strictly below the minimum observed WSE. When `--force_z0` is provided, the per-station `zmin` value from the CSV is used instead.

**Bayesian inference.** Parameters are estimated using a Metropolis MCMC sampler (4 chains, 10 000 sampling steps, 2 000 tuning steps) via [PyMC](https://www.pymc.io/). Weakly informative priors are:

| Parameter | Prior |
|---|---|
| *a* | TruncatedNormal(μ = 800, σ = 600, lower = 0.1) |
| *b* | TruncatedNormal(μ = 1.67, σ = 1.2, lower = 1) |
| *z*₀ | TruncatedNormal(μ = hmin − 5, σ = 30, upper = *z*₀\_constraint − 0.05) |

**Calibration strategy.** If the number of valid overlap pairs is ≥ `min_points` (default 12), the Bayesian fitter runs on observed data (**overlap approach**). Otherwise, it falls back to a **quantile approach**: year-monthly means are computed for both WSE and Q, 21 quantile levels are sampled from each, and paired by rank to produce 21 synthetic calibration pairs.

**Validation.** Performance metrics (NSE, KGE, PBIAS, NRMSE, R²) are computed between climate monthly means (summary) and year-monthly means (validation) of *Q*RC and *Q*insitu.

---

## Data Sources

- **WSE** — [USDA HydroWeb](https://hydroweb.next.theia-land.fr/), Jason-3 / Sentinel-6 altimetry products
- **Discharge** — [Global Runoff Data Centre (GRDC)](https://www.bafg.de/GRDC/), national hydrological services

---

## Citation

If you use this software in academic or research work, please cite:

> Paris, A. et al. (2025). *Bayesian Rating Curve Estimation from Satellite Altimetry and In-Situ Discharge* [Software]. Hydro Matters.

---

## License

This software is distributed under a permissive open-source licence. You are free to use, modify, and redistribute it, provided that the original authors and Hydro Matters are credited in any resulting publications or derivative works. See [LICENSE](LICENSE) for full terms.

---

## Contact

For questions or bug reports, please open an [issue](../../issues) or contact [adrien.paris@hydro-matters.fr](mailto:adrien.paris@hydro-matters.fr).
