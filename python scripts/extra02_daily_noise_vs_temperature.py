"""
extra02_daily_noise_vs_temperature.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Extra Analysis (outside the main pipeline)
Tests whether daily strain noise/fluctuation is attributable to daily
temperature cycles, per sensor.

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors:
  1. Loads Step 1's combined master file.
  2. Excludes the confirmed cabling-change window (21-23 Jul 2026), same as
     other extra-analysis scripts, so it doesn't distort the comparison.
  3. Computes, per calendar day: the day's strain range (max-min) and the
     day's exhibition temperature range (max-min).
  4. Plots the two daily-range time series together (dual Y-axis) for each
     sensor, so you can see by eye whether they track together.
  5. Computes the Pearson correlation between daily strain range and daily
     temperature range, per sensor - a high positive correlation means
     daily noise is largely temperature-driven.

OUTPUTS (saved to Output\\Round 2\\Extra Analysis Steps\\):
  daily_noise_vs_temperature.html   - one panel per sensor, daily ranges
  daily_noise_vs_temperature.csv    - correlation r, p-value, n per sensor

USAGE
=====
    Run AFTER step1_collation_alignment.py has produced its master file.
        python extra02_daily_noise_vs_temperature.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds - save
edits as extra02_daily_noise_vs_temperature_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION — edit this section only
# ══════════════════════════════════════════════════════════════════════════

BASE_DIR = (
    r"C:\Users\c2051290\OneDrive - Newcastle University\NCLU\PhD\My PhD"
    r"\01 Onsite Work and Engineering\01 Sensors\02 Canberra analysis"
    r"\06 Files for Python Final Round"
)

OUTPUT_BASE  = "Output"
ROUND_NAME   = "Round 2"
STEP1_FOLDER = "Step 1 Collation Alignment and Raw Plots"
STEP_FOLDER  = "Extra Analysis Steps"

FONT = "Arial"

# Sensor subfolder -> (display name, column name in master file, plot colour)
SENSOR_MAP = {
    "NP": ("Nose_Portside",    "Nose_Portside_Strain(\u00b5S)",    "#2196F3"),
    "NS": ("Nose_Starboard",   "Nose_Starboard_Strain(\u00b5S)",   "#FF9800"),
    "PB": ("Portside_Bottom",  "Portside_Bottom_Strain(\u00b5S)",  "#4CAF50"),
    "PT": ("Portside_Top",     "Portside_Top_Strain(\u00b5S)",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "Starboard_Bottom_Strain(\u00b5S)", "#F44336"),
    "ST": ("Starboard_Top",    "Starboard_Top_Strain(\u00b5S)",    "#8B4513"),
}

TEMP_COL = "Exhibition_Temp(\u00b0C)"
TEMP_COLOUR = "#555555"

# Confirmed cabling-change window - excluded before computing daily ranges.
EXCLUDE_RANGES = [
    ("2026-07-21 00:00", "2026-07-23 17:00"),
]

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


def main():
    section("Extra Analysis: Daily Noise vs Temperature")

    step1_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP1_FOLDER)
    master_path = safe_path(os.path.join(step1_dir, "All_Sensors_Aligned_Filled_Trimmed.csv"))

    if not os.path.exists(master_path):
        warn(f"Master file not found: {master_path}")
        warn("Run step1_collation_alignment.py first, or check ROUND_NAME.")
        return

    df = pd.read_csv(master_path, parse_dates=["Timestamp"])
    ok(f"Loaded master file: {len(df)} rows")

    if TEMP_COL not in df.columns:
        warn(f"Temperature column '{TEMP_COL}' not found in master file. Aborting.")
        return

    for start_str, end_str in EXCLUDE_RANGES:
        start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
        mask = (df["Timestamp"] >= start) & (df["Timestamp"] <= end)
        n_excluded = mask.sum()
        if n_excluded:
            df = df.loc[~mask].copy()
            ok(f"Excluded {n_excluded} rows in cabling-change window {start} to {end}")

    df["Date"] = df["Timestamp"].dt.date

    daily_temp = df.groupby("Date")[TEMP_COL].agg(lambda x: x.max() - x.min())
    daily_temp.name = "Temp_Daily_Range"

    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    try:
        os.makedirs(safe_path(out_dir), exist_ok=True)
    except PermissionError:
        warn(f"Could not create output folder (permission denied): {out_dir}")
        return

    valid_sensors = [(code, name, col, colour) for code, (name, col, colour) in SENSOR_MAP.items()
                      if col in df.columns]
    missing = [col for _, (name, col, colour) in SENSOR_MAP.items() if col not in df.columns]
    for col in missing:
        warn(f"Column '{col}' not found in master file - skipping.")

    n_sensors = len(valid_sensors)
    fig = make_subplots(
        rows=n_sensors, cols=1, shared_xaxes=True,
        subplot_titles=[name for _, name, _, _ in valid_sensors],
        specs=[[{"secondary_y": True}]] * n_sensors,
        vertical_spacing=0.04,
    )

    summary_rows = []

    for i, (code, sensor_name, col_name, colour) in enumerate(valid_sensors, start=1):
        section(sensor_name)
        d = df.dropna(subset=[col_name])
        daily_strain = d.groupby("Date")[col_name].agg(lambda x: x.max() - x.min())
        daily_strain.name = "Strain_Daily_Range"

        merged = pd.concat([daily_strain, daily_temp], axis=1).dropna()
        if len(merged) < 3:
            warn(f"{sensor_name}: insufficient daily data for correlation - skipped.")
            continue

        r, p = stats.pearsonr(merged["Temp_Daily_Range"], merged["Strain_Daily_Range"])
        ok(f"{sensor_name}: r={r:.3f}  p={p:.4g}  n_days={len(merged)}")

        summary_rows.append({
            "Sensor": sensor_name, "Pearson_r": r, "p_value": p, "n_days": len(merged),
        })

        dates = pd.to_datetime(merged.index)

        fig.add_trace(
            go.Scatter(x=dates, y=merged["Strain_Daily_Range"], mode="lines",
                       line=dict(color=colour, width=1.5),
                       name=f"{sensor_name} strain range", showlegend=False),
            row=i, col=1, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=dates, y=merged["Temp_Daily_Range"], mode="lines",
                       line=dict(color=TEMP_COLOUR, width=1.5, dash="dot"),
                       name="Temp range", showlegend=(i == 1)),
            row=i, col=1, secondary_y=True,
        )
        fig.update_yaxes(title_text="Strain range", row=i, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Temp range", row=i, col=1, secondary_y=True)

    fig.update_layout(
        font=dict(family=FONT, size=11),
        height=280 * n_sensors,
        legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
        margin=dict(l=65, r=60, t=60, b=40),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")

    html_path = safe_path(os.path.join(out_dir, "daily_noise_vs_temperature.html"))
    try:
        fig.write_html(html_path, include_plotlyjs="cdn")
        ok(f"Saved combined plot: {html_path}")
    except PermissionError:
        warn(f"Could not write plot (file open elsewhere?): {html_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = safe_path(os.path.join(out_dir, "daily_noise_vs_temperature.csv"))
        try:
            summary_df.to_csv(csv_path, index=False)
            ok(f"Saved correlation summary: {csv_path}")
        except PermissionError:
            warn(f"Could not write summary CSV (file open in Excel?): {csv_path}")

    section("Done")


if __name__ == "__main__":
    main()
