"""
step8_indoor_vs_outdoor_temperature.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 8 (standalone, NOT part of run_full_pipeline.py)
Compares indoor exhibition temperature against the outdoor weather station
reading above the same building, as a first look at whether/how the
building buffers or lags outdoor temperature swings.

WHY THIS EXISTS
================
Raised in a supervisor discussion about a possible SECOND pathway for
temperature to affect strain readings: outdoor temperature changes could
flex the building structure itself (independent of the aircraft's own
temperature), which could then transmit through the suspension rigging to
the aircraft. This script is the first, simplest check in that line of
enquiry — how does indoor temperature actually relate to outdoor? — before
attempting to test whether that pathway shows up in the strain data itself.
This script does NOT test the strain link; it only compares the two
temperature series.

NOT part of the main pipeline
===============================
This is a standalone, one-off analysis script. It is deliberately NOT
wired into run_full_pipeline.py.

TIMESTAMP HANDLING — PLEASE VERIFY
====================================
The outside file's timestamps carry an explicit "BST"/"GMT" suffix (British
local clock time, e.g. "15-May-25 12:00:00 PM BST"). This script strips
that suffix and parses the remainder as a naive local timestamp, on the
assumption that Step 1's indoor Exhibition_Temp timestamps are ALSO plain
local wall-clock time with no DST/UTC handling (consistent with how the
main pipeline treats all its timestamps). If the indoor timestamps are
instead UTC or some other convention, the two series would be offset by up
to an hour during BST periods — worth double-checking this assumption
before trusting results near the exact hour-of-day level (the Step 4c-style
diurnal timing question), though it won't materially affect a same-day
range/correlation comparison like the one this script does.

Outdoor readings are not exactly on the 15-minute grid (e.g. "12:15:01"
instead of "12:15:00"), so alignment uses merge_asof with an 8-minute
tolerance — matching the SNAP_TOLERANCE_MIN convention already used
elsewhere in this project for near-grid timestamps.

WHAT THIS SCRIPT DOES
======================
  1. Loads Step 1's indoor Exhibition_Temp from the master file.
  2. Loads the outdoor weather station CSV, parses its BST/GMT timestamps.
  3. Aligns the two series (nearest match, 8-minute tolerance).
  4. Plots both full time series together, on one graph.
  5. Computes each day's temperature range (max-min) for both indoor and
     outdoor, plots them together, and reports:
       - the Pearson correlation between daily indoor range and daily
         outdoor range (how closely indoor tracks outdoor's day-to-day
         swings)
       - the mean ratio of indoor range to outdoor range (a simple damping
         factor — how much smaller indoor swings are than outdoor's)
  6. Runs a simple lag check: correlates indoor temperature against
     outdoor temperature shifted by 0 to +6 hours, to see whether indoor
     more closely follows a DELAYED version of outdoor (consistent with
     building thermal mass / lag) rather than the same-instant reading.

OUTPUTS (saved to Output\\Round 2\\Step 8 Indoor vs Outdoor Temperature\\):
  indoor_vs_outdoor_temperature.html   - full time series, both temps overlaid
  daily_range_comparison.html          - daily range, both temps overlaid
  lag_correlation.html                 - correlation vs lag (0-6 hours)
  indoor_vs_outdoor_summary.csv        - range correlation, damping ratio, best lag

USAGE
=====
    Run AFTER step1_collation_alignment.py has produced its master file.
        python step8_indoor_vs_outdoor_temperature.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds - save
edits as step8_indoor_vs_outdoor_temperature_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION — edit this section only
# ══════════════════════════════════════════════════════════════════════════

BASE_DIR = (
    r"C:\Users\c2051290\OneDrive - Newcastle University\NCLU\PhD\My PhD"
    r"\01 Onsite Work and Engineering\01 Sensors\02 Canberra analysis"
    r"\06 Files for Python Final Round"
)

OUTPUT_BASE   = "Output"
ROUND_NAME    = "Round 2"
STEP1_FOLDER  = "Step 1 Collation Alignment and Raw Plots"
STEP_FOLDER   = "Step 8 Indoor vs Outdoor Temperature"

OUTSIDE_TEMP_DIR  = "Outside Enviornment"   # matches actual folder name (as given)
OUTSIDE_TEMP_FILE = "Outside Temp.csv"      # confirmed via `dir` - space, not underscore

FONT = "Arial"
INDOOR_TEMP_COL  = "Exhibition_Temp(\u00b0C)"
OUTSIDE_VALUE_COL = "Value (\u00b0C)"

INDOOR_COLOUR  = "#2196F3"
OUTSIDE_COLOUR = "#FF9800"

SNAP_TOLERANCE_MIN = 8   # matches project convention for near-grid timestamp matching

# ══════════════════════════════════════════════════════════════════════════
# END USER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════


def section(msg):
    print(f"\n{'-'*70}\n  {msg}\n{'-'*70}")


def ok(msg):
    print(f"  OK  {msg}")


def warn(msg):
    print(f"  !!  {msg}")


def safe_path(path):
    """Bypass Windows' 260-char MAX_PATH limit for deep OneDrive paths."""
    if os.name == "nt":
        abspath = os.path.abspath(path)
        if not abspath.startswith("\\\\?\\"):
            return "\\\\?\\" + abspath
    return path


def parse_outside_timestamp(series):
    """Strip trailing 'BST'/'GMT' suffix and parse as naive local time."""
    stripped = series.str.replace(r"\s+(BST|GMT)$", "", regex=True)
    return pd.to_datetime(stripped, format="%d-%b-%y %I:%M:%S %p", errors="coerce")


def daily_range(series_by_date):
    return series_by_date.max() - series_by_date.min()


def main():
    section("Step 8: Indoor vs Outdoor Temperature")

    # ── Load indoor temperature (Step 1 master) ──────────────────────────
    step1_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP1_FOLDER)
    master_path = safe_path(os.path.join(step1_dir, "All_Sensors_Aligned_Filled_Trimmed.csv"))
    if not os.path.exists(master_path):
        warn(f"Indoor master file not found: {master_path}")
        warn("Run step1_collation_alignment.py first, or check ROUND_NAME.")
        return

    indoor = pd.read_csv(master_path, parse_dates=["Timestamp"])
    if INDOOR_TEMP_COL not in indoor.columns:
        warn(f"Indoor temperature column '{INDOOR_TEMP_COL}' not found. Aborting.")
        return
    indoor = indoor[["Timestamp", INDOOR_TEMP_COL]].dropna().sort_values("Timestamp").reset_index(drop=True)
    ok(f"Loaded indoor temperature: {len(indoor)} rows")

    # ── Load outdoor temperature ──────────────────────────────────────────
    outside_path = safe_path(os.path.join(BASE_DIR, OUTSIDE_TEMP_DIR, OUTSIDE_TEMP_FILE))
    if not os.path.exists(outside_path):
        warn(f"Outdoor temperature file not found: {outside_path}")
        return

    outside_raw = pd.read_csv(outside_path, encoding="utf-8-sig")
    if "Timestamp" not in outside_raw.columns or OUTSIDE_VALUE_COL not in outside_raw.columns:
        warn(f"Expected columns 'Timestamp' and '{OUTSIDE_VALUE_COL}' not found "
             f"(has {list(outside_raw.columns)}). Aborting.")
        return

    outside_raw["Timestamp"] = parse_outside_timestamp(outside_raw["Timestamp"])
    n_unparsed = outside_raw["Timestamp"].isna().sum()
    if n_unparsed:
        warn(f"{n_unparsed} outdoor timestamps failed to parse and were dropped.")
    outside = outside_raw.dropna(subset=["Timestamp", OUTSIDE_VALUE_COL]).copy()
    outside = outside.sort_values("Timestamp").reset_index(drop=True)
    outside = outside.rename(columns={OUTSIDE_VALUE_COL: "Outside_Temp(\u00b0C)"})
    ok(f"Loaded outdoor temperature: {len(outside)} rows "
       f"({outside['Timestamp'].min()} to {outside['Timestamp'].max()})")

    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    try:
        os.makedirs(safe_path(out_dir), exist_ok=True)
    except PermissionError:
        warn(f"Could not create output folder (permission denied): {out_dir}")
        return

    # ── Align on nearest timestamp within tolerance ──────────────────────
    merged = pd.merge_asof(
        indoor, outside, on="Timestamp", direction="nearest",
        tolerance=pd.Timedelta(minutes=SNAP_TOLERANCE_MIN),
    ).dropna(subset=["Outside_Temp(\u00b0C)"])
    ok(f"Aligned {len(merged)} rows within {SNAP_TOLERANCE_MIN}-minute tolerance")

    if len(merged) < 10:
        warn("Too few aligned rows to proceed. Check timestamp parsing / overlap window.")
        return

    # ── Plot 1: full time series overlay ─────────────────────────────────
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=merged["Timestamp"], y=merged[INDOOR_TEMP_COL], mode="lines",
                               line=dict(color=INDOOR_COLOUR, width=1), name="Indoor (Exhibition)"))
    fig1.add_trace(go.Scatter(x=merged["Timestamp"], y=merged["Outside_Temp(\u00b0C)"], mode="lines",
                               line=dict(color=OUTSIDE_COLOUR, width=1), name="Outdoor"))
    fig1.update_layout(
        font=dict(family=FONT, size=12), yaxis_title="Temperature (\u00b0C)",
        legend=dict(orientation="h", y=-0.15, x=0, xanchor="left"),
        margin=dict(l=65, r=40, t=40, b=80), plot_bgcolor="white",
    )
    fig1.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig1.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    html1 = safe_path(os.path.join(out_dir, "indoor_vs_outdoor_temperature.html"))
    try:
        fig1.write_html(html1, include_plotlyjs="cdn")
        ok(f"Saved: {os.path.basename(html1)}")
    except PermissionError:
        warn(f"Could not write plot (file open elsewhere?): {html1}")

    # ── Daily range comparison ───────────────────────────────────────────
    merged["Date"] = merged["Timestamp"].dt.date
    daily_indoor = merged.groupby("Date")[INDOOR_TEMP_COL].agg(daily_range)
    daily_outside = merged.groupby("Date")["Outside_Temp(\u00b0C)"].agg(daily_range)
    daily = pd.concat([daily_indoor, daily_outside], axis=1)
    daily.columns = ["Indoor_Daily_Range", "Outdoor_Daily_Range"]
    daily = daily.dropna()

    r_range, p_range = stats.pearsonr(daily["Outdoor_Daily_Range"], daily["Indoor_Daily_Range"])
    damping_ratio = float((daily["Indoor_Daily_Range"] / daily["Outdoor_Daily_Range"]).replace(
        [np.inf, -np.inf], np.nan).dropna().mean())
    ok(f"Daily range correlation (indoor vs outdoor): r={r_range:.3f}  p={p_range:.4g}  n_days={len(daily)}")
    ok(f"Mean damping ratio (indoor range / outdoor range): {damping_ratio:.3f}")

    dates = pd.to_datetime(daily.index)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dates, y=daily["Indoor_Daily_Range"], mode="lines",
                               line=dict(color=INDOOR_COLOUR, width=1.5), name="Indoor daily range"))
    fig2.add_trace(go.Scatter(x=dates, y=daily["Outdoor_Daily_Range"], mode="lines",
                               line=dict(color=OUTSIDE_COLOUR, width=1.5, dash="dot"), name="Outdoor daily range"))
    fig2.update_layout(
        font=dict(family=FONT, size=12), yaxis_title="Daily temperature range (\u00b0C)",
        legend=dict(orientation="h", y=-0.15, x=0, xanchor="left"),
        margin=dict(l=65, r=40, t=40, b=80), plot_bgcolor="white",
    )
    fig2.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig2.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    html2 = safe_path(os.path.join(out_dir, "daily_range_comparison.html"))
    try:
        fig2.write_html(html2, include_plotlyjs="cdn")
        ok(f"Saved: {os.path.basename(html2)}")
    except PermissionError:
        warn(f"Could not write plot (file open elsewhere?): {html2}")

    # ── Lag check: does indoor follow a DELAYED outdoor more closely? ──
    ts_indexed = merged.set_index("Timestamp")
    lag_results = []
    for lag_hours in range(0, 7):
        shifted_outside = ts_indexed["Outside_Temp(\u00b0C)"].shift(freq=pd.Timedelta(hours=lag_hours))
        aligned = pd.concat([ts_indexed[INDOOR_TEMP_COL], shifted_outside], axis=1).dropna()
        aligned.columns = ["indoor", "outside_shifted"]
        if len(aligned) < 30:
            continue
        r, p = stats.pearsonr(aligned["outside_shifted"], aligned["indoor"])
        lag_results.append({"Lag_Hours": lag_hours, "r": r, "p_value": p, "n": len(aligned)})
        ok(f"Lag {lag_hours}h: r={r:.4f}  n={len(aligned)}")

    best_lag = None
    if lag_results:
        lag_df = pd.DataFrame(lag_results)
        best_lag = lag_df.loc[lag_df["r"].idxmax()]
        ok(f"Best-matching lag: {int(best_lag['Lag_Hours'])}h (r={best_lag['r']:.4f})")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=lag_df["Lag_Hours"], y=lag_df["r"], mode="lines+markers",
                                   line=dict(color=INDOOR_COLOUR, width=2)))
        fig3.update_layout(
            font=dict(family=FONT, size=12),
            xaxis_title="Outdoor temperature lag (hours)", yaxis_title="Correlation with indoor temperature",
            margin=dict(l=65, r=40, t=40, b=60), plot_bgcolor="white",
        )
        fig3.update_xaxes(showgrid=True, gridcolor="#eeeeee")
        fig3.update_yaxes(showgrid=True, gridcolor="#eeeeee")
        html3 = safe_path(os.path.join(out_dir, "lag_correlation.html"))
        try:
            fig3.write_html(html3, include_plotlyjs="cdn")
            ok(f"Saved: {os.path.basename(html3)}")
        except PermissionError:
            warn(f"Could not write plot (file open elsewhere?): {html3}")

    # ── Summary CSV ───────────────────────────────────────────────────────
    summary = {
        "Daily_Range_r": r_range, "Daily_Range_p": p_range, "n_days": len(daily),
        "Mean_Damping_Ratio_Indoor_over_Outdoor": damping_ratio,
    }
    if best_lag is not None:
        summary["Best_Lag_Hours"] = int(best_lag["Lag_Hours"])
        summary["Best_Lag_r"] = best_lag["r"]

    csv_path = safe_path(os.path.join(out_dir, "indoor_vs_outdoor_summary.csv"))
    try:
        pd.DataFrame([summary]).to_csv(csv_path, index=False)
        ok(f"Saved: {os.path.basename(csv_path)}")
    except PermissionError:
        warn(f"Could not write summary CSV (file open in Excel?): {csv_path}")

    section("Done")


if __name__ == "__main__":
    main()
