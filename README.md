# Bayesian Rating Curve Estimation from Satellite Altimetry and In-Situ Discharge

## Overview

This repository provides a standalone Python tool for estimating **power-law rating curves** from paired Water Surface Elevation (WSE) and river discharge (Q) time series. The approach combines Bayesian Markov Chain Monte Carlo (MCMC) inference with a physically constrained zero-flow datum (*z*₀), enabling uncertainty-aware parameter estimation suitable for large-sample hydrological studies.

**Two discharge datasets are used with distinct roles:**

- **Q_model** — Simulated discharge from the [RAPID/RRR routing model](https://doi.org/10.5281/zenodo.5519672) (David et al., 2021). Used exclusively for **rating curve calibration**.
- **Q_obs** — Observed discharge from gauge networks (GRDC, ANA, SCHAPI). Used exclusively for **validation** of the calibrated rating curve. Optional.

Both datasets follow the same file format and naming convention. All outputs are structured CSV files — no plots are generated.

---

## Rating Curve Model

$$Q = a \cdot (h - z_0)^{b}$$

where *Q* is river discharge (m³ s⁻¹), *h* is water surface elevation (m), *z*₀ is the effective zero-flow datum (m), and *a*, *b* are shape parameters. Parameters are estimated as posterior medians from a Bayesian MCMC sampler, and uncertainty is reported as posterior standard deviations.

---

## Repository Structure

```
.
├── compute_arc.py           # Main script — Bayesian RC estimation
├── environment.yml          # conda environment definition (arc)
├── setup_env.sh             # one-command environment setup helper
├── README.md                # This file
├── data/
    ├── WSE_clean/                 # Water Surface Elevation time series
    │   └── WSE_{BASIN}_{STATION}[_suffix].txt
    ├── Q_model/             # RAPID/RRR model discharge (calibration)
    │   └── Q_{BASIN}_{STATION}[_suffix].txt
    ├── Q_obs/               # Observed discharge — GRDC, ANA, SCHAPI (validation)
    │   └── Q_{BASIN}_{STATION}[_suffix].txt
    └── Zmin.csv             # Optional per-station z0 constraint table

```

---

## Input Data Format

All WSE and Q files (model and observed) are **semicolon-delimited** with a mandatory header.

### File naming convention

| Type | Pattern | Example |
|---|---|---|
| WSE | `WSE_{BASIN}_{STATION}[_suffix].txt` | `WSE_NIGER_MALANVILLE_JASON3.txt` |
| Q (model or obs) | `Q_{BASIN}_{STATION}[_suffix].txt` | `Q_NIGER_MALANVILLE.txt` |

WSE files are paired with Q files automatically by **bidirectional longest-prefix match** on the `{BASIN}_{STATION}` token. Either file may carry an extra suffix the other does not. Pairing is applied independently to Q_model and Q_obs folders.

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
| `source` | — | Data provider (e.g. `grdc`, `SCHAPI`, `RAPID`, `Jason3`) |

### Zmin.csv — per-station z0 constraint

An optional CSV file to supply a pre-defined *z*₀ upper constraint per station (e.g. from bathymetric surveys). Format:

```
station;lon;lat;zmin
NIGER_MALANVILLE;2.43;11.87;176.32
DANUBE_MOHACS;18.69;45.99;74.10
```

The `station` key must match `{BASIN}_{STATION}` (case-insensitive). If a station is absent, `min(WSE)` is used as fallback.

> **Note on temporal coverage.** WSE and Q series do not need to span the same period. Calibration uses the temporal overlap between WSE and Q_model. Q_obs is used independently for validation.

---

## Environment Setup

```bash
# Create the environment
bash setup_env.sh        # or: conda env create --name arc --file environment.yml

# Activate
conda activate arc

# Update an existing environment
bash setup_env.sh --update
```

Key dependencies: Python 3.11, NumPy ≥ 1.24, pandas ≥ 2.0, SciPy ≥ 1.11, PyMC ≥ 5.0, ArviZ ≥ 0.17, geopy ≥ 2.4. Requires Miniconda or Anaconda ≥ 23.x.

---

## Usage

```bash
python compute_arc.py  <WSE_folder>  <Q_model_folder>  [options]
```

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `WSE_folder` | required | Folder containing `WSE*.txt` files |
| `Q_model_folder` | required | Folder containing `Q*.txt` files (RAPID/RRR model — calibration) |
| `-qobs / --Q_obs_folder` | `None` | Folder containing observed `Q*.txt` files (GRDC/ANA/SCHAPI — validation only). If omitted, no `validation.csv` is produced. |
| `-z0 / --force_z0` | `None` | Path to `Zmin.csv` (`station;lon;lat;zmin`) for per-station *z*₀ constraints |
| `-o / --outpath` | `WSE_folder/rating_curve/` | Output directory |
| `-b / --basin` | `None` | Restrict to files matching a basin token (e.g. `NIGER`) |
| `-m / --min_points` | `12` | Minimum overlap pairs before falling back to the quantile approach |

### Examples

```bash
# Calibration only — no validation
python compute_arc.py ./data/WSE ./data/Q_model

# Calibration + validation against observed discharge
python compute_arc.py ./data/WSE ./data/Q_model -qobs ./data/Q_obs

# With per-station z0 constraints and custom output folder
python compute_arc.py ./data/WSE ./data/Q_model -qobs ./data/Q_obs \
    -z0 ./data/Zmin.csv -o ./results

# Niger basin only
python compute_arc.py ./data/WSE ./data/Q_model -qobs ./data/Q_obs -b NIGER
```

---

## Outputs

### `rating_curve_summary.csv` — one row per station

| Column | Description |
|---|---|
| `basin`, `station` | Station identifiers |
| `lon`, `lat` | Coordinates (WGS84) |
| `a`, `b`, `z0` | RC parameters — posterior medians |
| `a_sd`, `b_sd`, `z0_sd` | Posterior standard deviations |
| `zmin` | Minimum WSE of the calibration data (m) |
| `NSE`, `KGE`, `PBIAS`, `NRMSE`, `R2` | Performance on climate monthly means (Q_model vs Q_RC) |
| `approach` | Calibration strategy: `overlap` or `quantile` |
| `nb_cal_points` | Number of pairs used for fitting |
| `nb_clim_months` | Number of climate months used for summary statistics |

### `validation.csv` — one row per station (requires `--Q_obs_folder`)

Statistics computed on **year-monthly means** of *Q*RC versus *Q*obs over their common temporal overlap.

| Column | Description |
|---|---|
| `basin`, `station`, `lon`, `lat` | Station identifiers and coordinates |
| `NSE`, `KGE`, `PBIAS`, `NRMSE`, `R2` | Performance metrics (Q_RC vs Q_obs) |
| `val_start`, `val_end` | First and last `YYYY-MM` of the validation period |
| `nb_months` | Number of year-months compared |

---

## Methods

**Date matching.** WSE and Q_model observations are paired if timestamps are within ±24 hours of each other (closest neighbour).

**Outlier filtering.** A local monotonicity filter in WSE-sorted order flags observations whose discharge falls more than 20 % below the median discharge of their 5 nearest WSE-space neighbours.

**z₀ constraint.** The zero-flow datum is constrained to remain strictly below the minimum observed WSE. When `--force_z0` is provided, the per-station `zmin` from `Zmin.csv` is used instead.

**Bayesian inference.** Parameters estimated using a Metropolis MCMC sampler (4 chains, 10 000 sampling steps, 2 000 tuning steps) via [PyMC](https://www.pymc.io/). Weakly informative priors:

| Parameter | Prior |
|---|---|
| *a* | TruncatedNormal(μ = 800, σ = 600, lower = 0.1) |
| *b* | TruncatedNormal(μ = 1.67, σ = 1.2, lower = 1) |
| *z*₀ | TruncatedNormal(μ = hmin − 5, σ = 30, upper = *z*₀\_constraint − 0.05) |

**Calibration strategy.** If the number of valid WSE–Q_model overlap pairs is ≥ `min_points` (default 12), the Bayesian fitter runs on observed overlap data (**overlap approach**). Otherwise, it falls back to a **quantile approach**: year-monthly means are computed for both WSE and Q_model, 21 quantile levels are sampled from each and paired by rank to produce 21 synthetic calibration pairs.

**Validation.** When Q_obs data are provided, performance metrics (NSE, KGE, PBIAS, NRMSE, R²) are computed between year-monthly means of *Q*RC and *Q*obs over their common temporal window.

---

## Data Sources

- **WSE** — [HydroWeb](https://hydroweb.next.theia-land.fr/), Jason-3 / Sentinel-6 altimetry products
- **Q_model** — [RAPID/RRR routing model](https://doi.org/10.5281/zenodo.5519672) (David et al., 2021)
- **Q_obs** — [GRDC](https://www.bafg.de/GRDC/), [ANA](https://www.gov.br/ana/), [SCHAPI](https://www.hydro.eaufrance.fr/)

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
