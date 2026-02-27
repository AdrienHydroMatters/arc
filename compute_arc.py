#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 08:36:32 2025

@author: Laetitia GAL (laetitia.gal@hydro-matters.fr)

Copyright (c) 2024 Hydro Matters

Permission is hereby granted, free of charge, to any person or entity obtaining
a copy of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

You may not claim ownership or authorship of this software.

If you use this software in any academic, research, or commercial work, you are
required to provide appropriate attribution to the author and company. This
includes but is not limited to mentioning the author's name and the company's 
name in any articles, publications, or documentation related to the use of this 
software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rating_curve.py
---------------
Bayesian Rating Curve Estimation from WSE and Discharge time series.

Computes the power-law rating curve  Q = a * (WSE - z0)^b
using Bayesian inference (PyMC / Metropolis sampler).

Calibration strategy
--------------------
    1. OVERLAP (preferred): date-matched WSE/Q pairs within +/-24 h.
       Used when valid overlap points >= min_points (default 12).
    2. QUANTILE (fallback): when overlap is insufficient.
       - Compute year-monthly means (one value per YYYY-MM) for both
         WSE and Q over the full available record.
       - Draw 21 quantile levels (0, 5, 10, ... 100%) from those
         year-monthly series and pair them by rank.
       - These 21 synthetic (WSE, Q) pairs feed the Bayesian fitter.

Validation statistics
---------------------
    Climate monthly means (Jan-Dec, averaged across all years) of
    Qrc (rating curve applied to WSE altimetry) are compared to Qinsitu
    climate monthly means using NSE, KGE, pBIAS, NRMSE, R2.
    A validation.csv is produced for stations where comparison is possible.

Outputs (in <outpath>/)
-----------------------
    rating_curve_summary.csv   one row per station:
                               basin, station, lon, lat,
                               a, b, z0, a_sd, b_sd, z0_sd, zmin,
                               NSE, KGE, PBIAS, NRMSE, R2,
                               approach, nb_points
    validation.csv             one row per station (when stats available):
                               basin, station, lon, lat,
                               NSE, KGE, PBIAS, NRMSE, R2,
                               val_start, val_end, nb_months

Usage
-----
    python rating_curve.py  <WSE_folder>  <Q_folder>  [options]

    positional arguments:
        WSE_folder          folder containing WSE*.txt files
        Q_folder            folder containing Q*.txt files

    optional arguments:
        -z0  --force_z0     str    Path to a semicolon-delimited CSV with columns
                                   station;zmin  where station = {BASIN}_{STATION}.
                                   If provided, the zmin value for each station is used
                                   as the z0 upper constraint instead of min(WSE).
                                   If a station is not found in the CSV, min(WSE) is used.
        -o   --outpath      str    Output folder. Default: WSE_folder/rating_curve/
        -b   --basin        str    Filter files by basin name (e.g. NIGER).
        -m   --min_points   int    Minimum overlap points before falling back to
                                   the quantile approach (default: 12).

Input file format (semicolon-delimited, header required)
---------------------------------------------------------
    station ; lon ; lat ; date               ; value ; uncertainty ; source
    NIGER_A ; 2.5 ; 11.8; 2010-01-15 12:00:00; 243.5 ; 0.1        ; Jason3

Dependencies
------------
    numpy, pandas, pymc, scipy, warnings
"""

import os
import glob
import argparse
import warnings
import logging

import numpy as np
import pandas as pd
import pymc as pm

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("pymc").setLevel(logging.ERROR)


# =============================================================================
# I/O HELPERS
# =============================================================================

def read_hm_df(filename):
    """Read a semicolon-delimited hydro-matters file into a DataFrame."""
    df = pd.read_csv(filename, sep=';', dtype=str)
    df.columns = df.columns.str.strip()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['date']  = df['date'].str.strip()
    return df


def append_row(row_dict, cols, outfile):
    """Append one row to a semicolon CSV, writing header on first call."""
    write_header = not os.path.exists(outfile)
    with open(outfile, 'a') as f:
        if write_header:
            f.write(';'.join(cols) + '\n')
        f.write(';'.join(str(row_dict.get(c, '')) for c in cols) + '\n')


# =============================================================================
# Z0 TABLE LOADER
# =============================================================================

def load_z0_table(csv_path):
    """
    Load a per-station z0 (zmin) constraint table.

    Expected format (semicolon-delimited, header required):
        station;lon;lat;zmin

    The station key must be {BASIN}_{STATION} (case-insensitive lookup).

    Parameters
    ----------
    csv_path : str  path to the CSV file, or None

    Returns
    -------
    dict  {station_key_upper: hmin_float}  or empty dict if csv_path is None
    """
    if csv_path is None:
        return {}
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    if 'station' not in df.columns or 'zmin' not in df.columns:
        raise ValueError(
            f"z0 CSV must have columns 'station' and 'zmin' (got: {list(df.columns)})"
        )
    table = {}
    for _, row in df.iterrows():
        key = str(row['station']).strip().upper()
        try:
            table[key] = float(row['zmin'])
        except (ValueError, TypeError):
            pass   # skip rows with non-numeric zmin
    print(f'  [z0 table] loaded {len(table)} station entries from {csv_path}')
    return table


# =============================================================================
# DATE MATCHING
# =============================================================================

def match_dates(dates_wse, dates_q, max_hours=24):
    """
    Return index pairs (idx_wse, idx_q) where timestamps are within
    max_hours of each other (closest-neighbour).
    Both inputs are arrays of strings 'YYYY-MM-DD HH:MM:SS'.
    """
    dt_wse = pd.to_datetime(dates_wse)
    dt_q   = pd.to_datetime(dates_q)
    idx_wse, idx_q = [], []
    for i, dw in enumerate(dt_wse):
        diffs = np.abs((dt_q - dw).total_seconds()) / 3600.0
        j = int(np.argmin(diffs))
        if diffs[j] <= max_hours:
            idx_wse.append(i)
            idx_q.append(j)
    return np.array(idx_wse, dtype=int), np.array(idx_q, dtype=int)


# =============================================================================
# OUTLIER FILTER
# =============================================================================

def filter_monotonic_local(dates, wse, Q, window=5, rel_drop=0.2):
    """
    Remove points where Q drops more than rel_drop below the local median
    in WSE-sorted order (catches gross stage-discharge inconsistencies).
    """
    dates = np.asarray(dates)
    wse   = np.asarray(wse,  dtype=float)
    Q     = np.asarray(Q,    dtype=float)

    order    = np.argsort(wse)
    q_sorted = Q[order]
    mask_s   = np.ones(len(q_sorted), dtype=bool)

    for i in range(len(q_sorted)):
        i0    = max(0, i - window)
        i1    = min(len(q_sorted), i + window + 1)
        q_med = np.median(q_sorted[i0:i1])
        if q_sorted[i] < (1 - rel_drop) * q_med:
            mask_s[i] = False

    mask = np.zeros(len(wse), dtype=bool)
    mask[order] = mask_s
    return dates[mask], wse[mask], Q[mask]


# =============================================================================
# TEMPORAL MEANS
# =============================================================================

def yearmonthly_mean(dates, values):
    """
    Year-monthly means: one value per YYYY-MM.
    Returns (unique_ym_strings_array, mean_values_array).
    Mirrors original yearmonthly_mean() from functions.py.
    """
    dt = pd.to_datetime(dates)
    df = pd.DataFrame({
        'ym':    dt.strftime('%Y-%m'),
        'value': np.array(values, dtype=float)
    })
    grouped = df.groupby('ym')['value'].mean().reset_index()
    return grouped['ym'].values, grouped['value'].values


def climate_monthly_mean(dates, values):
    """
    Climate monthly means: average across all years for each calendar
    month Jan=1 ... Dec=12.
    Mirrors original monthly_mean() from functions.py.
    Returns dict {month_int: mean_value}.
    """
    dt = pd.to_datetime(dates)
    df = pd.DataFrame({'month': dt.month,
                       'value': np.array(values, dtype=float)})
    grouped = df.groupby('month')['value'].mean()
    return {int(m): float(v) for m, v in grouped.items()}


# =============================================================================
# QUANTILE FALLBACK
# =============================================================================

def rc_quantile_pairs(wse_dates, wse_values, q_dates, q_values):
    """
    Build 21 synthetic (WSE, Q) calibration pairs via the quantile approach.

    Faithfully reproduces the original logic:
        wsem       = yearmonthly_mean(wse)[1]   # one value per YYYY-MM
        dischargem = yearmonthly_mean(q)[1]
        wse_q  = np.percentile(wsem,       np.arange(0, 101, 5))
        q_q    = np.percentile(dischargem, np.arange(0, 101, 5))

    The 21 quantile levels (0, 5, 10, ..., 100%) span the full
    hydrological range and are paired by rank.
    """
    _, wse_ym = yearmonthly_mean(wse_dates, wse_values)
    _, q_ym   = yearmonthly_mean(q_dates,   q_values)

    quant = np.arange(0, 101, 5)          # 21 levels
    wse_q = np.percentile(wse_ym, quant)
    q_q   = np.percentile(q_ym,   quant)

    return wse_q, q_q


# =============================================================================
# HYDROLOGICAL CRITERIA
# =============================================================================

def calc_criteria(obs, sim):
    """
    Compute NSE, KGE, PBIAS, NRMSE, R2 between obs and sim.
    NaN pairs are excluded. Returns all-NaN dict if < 2 valid pairs.
    """
    nan_result = {'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan,
                  'NRMSE': np.nan, 'R2': np.nan, 'nb_points': 0}

    obs = np.array(obs, dtype=float)
    sim = np.array(sim, dtype=float)
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[valid], sim[valid]

    if len(obs) < 2:
        return nan_result

    mean_obs = np.mean(obs)

    pbias = (np.sum(sim - obs) / np.sum(obs)) * 100.0

    rmse  = np.sqrt(np.mean((obs - sim) ** 2))
    r_obs = np.max(obs) - np.min(obs)
    nrmse = (rmse / r_obs) * 100.0 if r_obs > 0 else np.nan

    nse = 1.0 - (np.sum((sim - obs) ** 2) /
                 np.sum((obs - mean_obs) ** 2))

    rho   = np.corrcoef(obs, sim)[0, 1]
    beta  = np.mean(sim) / mean_obs
    gamma = (np.std(sim) / np.std(obs)) if np.std(obs) > 0 else np.nan
    kge   = 1.0 - np.sqrt((rho - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)

    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - mean_obs) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'NSE':       round(float(nse),   4),
        'KGE':       round(float(kge),   4),
        'PBIAS':     round(float(pbias),  4),
        'NRMSE':     round(float(nrmse),  4),
        'R2':        round(float(r2),     4),
        'nb_points': int(len(obs)),
    }


# =============================================================================
# RATING CURVE MODEL
# =============================================================================

def rating_curve(wse, a, b, z0):
    """Power-law rating curve: Q = a * (WSE - z0)^b.
    Returns NaN where WSE - z0 <= 0 to avoid invalid power."""
    wse  = np.asarray(wse, dtype=float)
    diff = wse - z0
    diff = np.where(diff > 0, diff, np.nan)
    return a * diff ** b


def rc_bayesian(wse, discharge, z0_constraint, n_steps=10_000):
    """
    Bayesian inference (Metropolis sampler) for a, b, z0.

    Parameters
    ----------
    wse            : 1-D array  water surface elevation [m]
    discharge      : 1-D array  observed discharge [m3/s]
    z0_constraint  : float      hard upper bound for z0
    n_steps        : int        MCMC sampling steps per chain

    Returns
    -------
    a, b, z0          : posterior medians
    a_sd, b_sd, z0_sd : posterior standard deviations
    zmin              : min(wse) of calibration data
    """
    zmin = float(np.min(wse))

    with pm.Model():
        a_rv  = pm.TruncatedNormal('a',  mu=800,      sigma=600,
                                   lower=0.1, initval=800)
        b_rv  = pm.TruncatedNormal('b',  mu=1.67,     sigma=1.2,
                                   lower=1,   initval=1.67)
        z0_rv = pm.TruncatedNormal('z0', mu=zmin - 5, sigma=30,
                                   upper=z0_constraint - 0.05,
                                   initval=zmin - 5)
        mu_q  = a_rv * (wse - z0_rv) ** b_rv
        sigma = pm.HalfNormal('sigma', sigma=1)
        pm.Normal('discharge_obs', mu=mu_q, sigma=sigma,
                  observed=discharge)
        step  = pm.Metropolis()
        trace = pm.sample(n_steps, tune=2000, chains=4,
                          step=step, return_inferencedata=True,
                          progressbar=False)

    a_s  = trace.posterior['a'].values.flatten()
    b_s  = trace.posterior['b'].values.flatten()
    z0_s = trace.posterior['z0'].values.flatten()

    return (float(np.median(a_s)),  float(np.median(b_s)),  float(np.median(z0_s)),
            float(np.std(a_s)),     float(np.std(b_s)),     float(np.std(z0_s)),
            zmin)


# =============================================================================
# OUTPUT COLUMN DEFINITIONS
# =============================================================================

SUMMARY_COLS = [
    'basin', 'station', 'lon', 'lat',
    'a', 'b', 'z0', 'a_sd', 'b_sd', 'z0_sd', 'zmin',
    'NSE', 'KGE', 'PBIAS', 'NRMSE', 'R2',
    'approach', 'nb_cal_points', 'nb_clim_months',
]
# nb_cal_points  : number of (WSE, Q) pairs used to fit the RC
#                  (overlap count, or 21 for quantile approach)
# nb_clim_months : number of common calendar months (1-12) used
#                  for the summary statistics (climate monthly means)

VALIDATION_COLS = [
    'basin', 'station', 'lon', 'lat',
    'NSE', 'KGE', 'PBIAS', 'NRMSE', 'R2',
    'val_start', 'val_end', 'nb_months',
]


# =============================================================================
# FILE-PAIRING HELPERS
# =============================================================================

def strip_type_prefix(path, prefix):
    """Strip leading type tag (WSE_ or Q_) and return stem without extension."""
    name = os.path.splitext(os.path.basename(path))[0]
    if name.upper().startswith(prefix.upper()):
        name = name[len(prefix):]
    return name


def shared_prefix_len(a, b):
    """
    Bidirectional prefix match (case-insensitive).
    Returns length of shared {BASIN}_{STATION} token, or 0 if none.
    The shorter stem must be a prefix of the longer, followed by '_' or end.
    """
    a_up, b_up = a.upper(), b.upper()
    short, long_ = (a_up, b_up) if len(a_up) <= len(b_up) else (b_up, a_up)
    if short == long_:
        return len(short)
    if long_.startswith(short) and (len(long_) == len(short) or
                                    long_[len(short)] == '_'):
        return len(short)
    return 0


def best_q_match(wse_stem, q_stem_map):
    """Return the Q file with the longest shared prefix with wse_stem."""
    best_file, best_len = None, 0
    for qf, qs in q_stem_map.items():
        n = shared_prefix_len(wse_stem, qs)
        if n > best_len:
            best_file, best_len = qf, n
    return best_file


def parse_basin_station(wse_file):
    """
    Extract basin (token 1) and station (token 2) from
    WSE_{BASIN}_{STATION}[_suffix].txt.
    """
    stem  = os.path.splitext(os.path.basename(wse_file))[0]
    parts = stem.split('_')
    basin   = parts[1] if len(parts) > 1 else ''
    station = parts[2] if len(parts) > 2 else ''
    return basin, station


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_station(wse_file, q_file, outpath, z0_table=None, min_points=12):
    """Compute the rating curve for one WSE / Q file pair and write outputs."""
    basin, station = parse_basin_station(wse_file)
    label = f'{basin}_{station}'
    print(f'\n--- Processing: {label} ---')

    # ── load ──────────────────────────────────────────────────────────────
    wse_df = read_hm_df(wse_file)
    q_df   = read_hm_df(q_file)

    wse_df = wse_df.dropna(subset=['value'])
    q_df   = q_df[q_df['value'] >= 0].dropna(subset=['value'])

    if wse_df.empty or q_df.empty:
        print('  [skip] empty file after cleaning')
        return

    lon = float(wse_df['lon'].iloc[0]) if 'lon' in wse_df.columns else np.nan
    lat = float(wse_df['lat'].iloc[0]) if 'lat' in wse_df.columns else np.nan

    # ── overlap date matching + outlier filter ─────────────────────────────
    idx_wse, idx_q = match_dates(wse_df['date'].values, q_df['date'].values)

    wse_ov = q_ov = np.array([])
    if len(idx_wse) > 0:
        dates_raw = wse_df['date'].values[idx_wse]
        wse_raw   = wse_df['value'].values[idx_wse].astype(float)
        q_raw     = q_df['value'].values[idx_q].astype(float)
        n_before  = len(wse_raw)
        _, wse_ov, q_ov = filter_monotonic_local(dates_raw, wse_raw, q_raw)
        print(f'  Overlap points: {n_before} -> {len(wse_ov)} after outlier filter')

    # ── calibration strategy ───────────────────────────────────────────────
    if len(wse_ov) >= min_points:
        approach  = 'overlap'
        wse_fit   = wse_ov
        q_fit     = q_ov
        nb_points = len(wse_fit)
        print(f'  Strategy: OVERLAP ({nb_points} points)')
    else:
        approach = 'quantile'
        print(f'  Only {len(wse_ov)} overlap points (need {min_points}). '
              'Falling back to QUANTILE approach.')
        wse_fit, q_fit = rc_quantile_pairs(
            wse_df['date'].values, wse_df['value'].values,
            q_df['date'].values,   q_df['value'].values,
        )
        nb_points = len(wse_fit)
        print(f'  Strategy: QUANTILE ({nb_points} synthetic pairs)')

    # ── z0 constraint ──────────────────────────────────────────────────────
    # Look up station key {BASIN}_{STATION} in the z0 table (case-insensitive).
    station_key = f'{basin}_{station}'.upper()
    z0_table = z0_table or {}
    if station_key in z0_table:
        z0_constraint = float(z0_table[station_key])
        z0_source     = 'from z0 CSV'
    else:
        z0_constraint = float(np.min(wse_fit))
        z0_source     = f'min WSE - {approach}'
    print(f'  z0 constraint = {z0_constraint:.4f} m [{z0_source}]')

    # ── Bayesian fitting ───────────────────────────────────────────────────
    print('  Running Bayesian MCMC ...')
    a, b, z0, a_sd, b_sd, z0_sd, zmin = rc_bayesian(
        wse_fit, q_fit, z0_constraint)
    print(f'  a={a:.4f}+/-{a_sd:.4f}  b={b:.4f}+/-{b_sd:.4f}  '
          f'z0={z0:.4f}+/-{z0_sd:.4f}')

    # ── apply RC to full WSE altimetry timeseries ─────────────────────────
    wse_all  = wse_df['value'].values.astype(float)
    q_rc_all = rating_curve(wse_all, a, b, z0)

    # ==========================================================
    # SUMMARY STATISTICS  —  climate monthly means (12 pts max)
    # ==========================================================
    # Average all observations within each calendar month (Jan=1..Dec=12)
    # independently for Qrc and Qinsitu, then compare the 12 values.
    clim_qrc  = climate_monthly_mean(wse_df['date'].values, q_rc_all)
    clim_qobs = climate_monthly_mean(q_df['date'].values,
                                     q_df['value'].values)

    common_clim = sorted(set(clim_qrc.keys()) & set(clim_qobs.keys()))
    nb_clim_months = len(common_clim)   # 0-12

    crit_summary = {'NSE': np.nan, 'KGE': np.nan, 'PBIAS': np.nan,
                    'NRMSE': np.nan, 'R2': np.nan}

    if nb_clim_months >= 2:
        qobs_clim = np.array([clim_qobs[m] for m in common_clim])
        qrc_clim  = np.array([clim_qrc[m]  for m in common_clim])
        crit_summary = calc_criteria(qobs_clim, qrc_clim)
        print(f'  Summary stats ({nb_clim_months} climate monthly means): '
              f'NSE={crit_summary["NSE"]:.3f}  '
              f'KGE={crit_summary["KGE"]:.3f}  '
              f'R2={crit_summary["R2"]:.3f}')

    # ── write rating_curve_summary.csv ─────────────────────────────────────
    os.makedirs(outpath, exist_ok=True)
    summary_row = {
        'basin':          basin,
        'station':        station,
        'lon':            round(lon, 4),
        'lat':            round(lat, 4),
        'a':              round(a,    4),
        'b':              round(b,    4),
        'z0':             round(z0,   4),
        'a_sd':           round(a_sd,  4),
        'b_sd':           round(b_sd,  4),
        'z0_sd':          round(z0_sd, 4),
        'zmin':           round(zmin,  4),
        'NSE':            crit_summary['NSE'],
        'KGE':            crit_summary['KGE'],
        'PBIAS':          crit_summary['PBIAS'],
        'NRMSE':          crit_summary['NRMSE'],
        'R2':             crit_summary['R2'],
        'approach':       approach,
        'nb_cal_points':  nb_points,
        'nb_clim_months': nb_clim_months,
    }
    append_row(summary_row, SUMMARY_COLS,
               os.path.join(outpath, 'rating_curve_summary.csv'))

    # ==========================================================
    # VALIDATION STATISTICS  —  year-monthly means (N pts)
    # ==========================================================
    # Compute one value per YYYY-MM for both Qrc and Qinsitu,
    # align on the common YYYY-MM keys, then compare.
    # This gives a variable number of points depending on the
    # length of the records and their temporal overlap.
    ym_qrc_keys,  ym_qrc_vals  = yearmonthly_mean(wse_df['date'].values,
                                                   q_rc_all)
    ym_qobs_keys, ym_qobs_vals = yearmonthly_mean(q_df['date'].values,
                                                   q_df['value'].values)

    ym_qrc_dict  = dict(zip(ym_qrc_keys,  ym_qrc_vals))
    ym_qobs_dict = dict(zip(ym_qobs_keys, ym_qobs_vals))

    common_ym  = sorted(set(ym_qrc_keys) & set(ym_qobs_keys))
    nb_val_months = len(common_ym)

    if nb_val_months >= 2:
        qobs_ym = np.array([ym_qobs_dict[m] for m in common_ym])
        qrc_ym  = np.array([ym_qrc_dict[m]  for m in common_ym])
        crit_val = calc_criteria(qobs_ym, qrc_ym)
        val_start = common_ym[0]
        val_end   = common_ym[-1]
        print(f'  Validation stats ({nb_val_months} year-monthly means, '
              f'{val_start} to {val_end}): '
              f'NSE={crit_val["NSE"]:.3f}  '
              f'KGE={crit_val["KGE"]:.3f}  '
              f'R2={crit_val["R2"]:.3f}')

        val_row = {
            'basin':      basin,
            'station':    station,
            'lon':        round(lon, 4),
            'lat':        round(lat, 4),
            'NSE':        crit_val['NSE'],
            'KGE':        crit_val['KGE'],
            'PBIAS':      crit_val['PBIAS'],
            'NRMSE':      crit_val['NRMSE'],
            'R2':         crit_val['R2'],
            'val_start':  val_start,
            'val_end':    val_end,
            'nb_months':  nb_val_months,
        }
        append_row(val_row, VALIDATION_COLS,
                   os.path.join(outpath, 'validation.csv'))

    print(f'  Done -> {label}')


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bayesian Rating Curve computation (no plots).',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('WSE_folder', help='Folder containing WSE*.txt files')
    parser.add_argument('Q_folder',   help='Folder containing Q*.txt files')
    parser.add_argument('-z0', '--force_z0',   default=None,
                        help='Path to a semicolon CSV with columns station;lon;lat;zmin\n'
                             'station key = {BASIN}_{STATION}.\n'
                             'If omitted, min(WSE) is used as z0 constraint.')
    parser.add_argument('-o',  '--outpath',    default=None,
                        help='Output folder (default: WSE_folder/rating_curve/)')
    parser.add_argument('-b',  '--basin',      default=None,
                        help='Filter by basin name (e.g. NIGER)')
    parser.add_argument('-m',  '--min_points', type=int, default=12,
                        help='Minimum overlap points before falling back to '
                             'the quantile approach (default: 12)')
    args = parser.parse_args()
    WSE_folder  = args.WSE_folder
    Q_folder    = args.Q_folder
    z0_csv      = args.force_z0   # path to CSV or None
    outpath     = args.outpath
    basin       = args.basin
    min_points  = args.min_points

    outpath  = outpath or os.path.join(WSE_folder, 'rating_curve')
    os.makedirs(outpath, exist_ok=True)

    z0_table = load_z0_table(z0_csv)

    # ── discover files ─────────────────────────────────────────────────────
    basin_tag = f'*_{basin.upper()}_*' if basin else '*'
    wse_files = sorted(glob.glob(os.path.join(WSE_folder, f'WSE{basin_tag}.txt')))
    q_files   = sorted(glob.glob(os.path.join(Q_folder,   f'Q{basin_tag}.txt')))

    if not wse_files:
        raise FileNotFoundError(f'No WSE*.txt files found in {WSE_folder}')
    if not q_files:
        raise FileNotFoundError(f'No Q*.txt files found in {Q_folder}')

    print(f'Found {len(wse_files)} WSE files and {len(q_files)} Q files')

    # ── pair WSE <-> Q by bidirectional longest-prefix match ──────────────
    # File convention:  WSE_{BASIN}_{STATION}[_suffix].txt
    #                   Q_{BASIN}_{STATION}[_suffix].txt
    # Either file may carry an extra suffix the other does not.
    # Examples:
    #   WSE_ADOUR_ADOUR-KM0115-EXP.txt  <->  Q_ADOUR_ADOUR-KM0115-EXP_rivid-23024083.txt
    #   WSE_NIGER_MALANVILLE_JASON3.txt  <->  Q_NIGER_MALANVILLE.txt
    wse_stems = {wf: strip_type_prefix(wf, 'WSE_') for wf in wse_files}
    q_stems   = {qf: strip_type_prefix(qf, 'Q_')   for qf in q_files}

    processed, skipped = 0, 0
    for wf in wse_files:
        wse_stem = wse_stems[wf]
        qf = best_q_match(wse_stem, q_stems)
        if qf is None:
            print(f'[no Q match] {os.path.basename(wf)} -> skipping')
            skipped += 1
            continue
        print(f'  Matched: {os.path.basename(wf)}  <->  {os.path.basename(qf)}')
        try:
            process_station(wf, qf, outpath,
                            z0_table=z0_table,
                            min_points=min_points)
            processed += 1
        except Exception as e:
            print(f'  [ERROR] {os.path.basename(wf)}: {e}')
            skipped += 1

    print(f'\n=== Finished: {processed} computed, {skipped} skipped ===')
    print(f'Output folder: {outpath}')


if __name__ == '__main__':
    main()