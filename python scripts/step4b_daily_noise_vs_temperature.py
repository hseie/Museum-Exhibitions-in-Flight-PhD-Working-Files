"""
step4b_daily_noise_vs_temperature.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 4b
Confirms that the shared summer-bigger/winter-smaller daily-oscillation
envelope visible across all six sensors' STL "Daily seasonal" panels is
explained by the seasonal envelope of daily temperature range.

WHERE THIS SITS
=================
Runs AFTER Step 4 (STL Decomposition), using its per-sensor decomposed
output. Placed as "4b" rather than renumbering Steps 5-7, since nothing
later in the pipeline depends on this script's output - it's a standalone
diagnostic, not an input to Step 5 onward.

SCOPE
======
Only the seasonal_daily component and temperature are compared here -
the residual component was checked in an earlier pass (see prior round's
Step 4b) and found to track temperature far more weakly and noisily, so
it's intentionally left out of this version's plotting and correlation.

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors:
  1. Loads Step 4's per-sensor decomposed CSV.
  2. Loads Step 1's master file for Exhibition_Temp, merged on Timestamp.
  3. Excludes the confirmed cabling-change window (21-23 Jul 2026).
  4. Computes, per calendar day: the day's temperature range, and the
     day's seasonal_daily range.
  5. Plots the two daily-range series together per sensor (dual Y-axis).
  6. Computes the Pearson correlation between daily seasonal_daily range
     and daily temperature range, per sensor.

OUTPUTS (saved to Output\\Round 2\\Step 4b Daily Noise vs Temperature\\):
  daily_seasonal_vs_temperature.html   - one panel per sensor
  daily_seasonal_vs_temperature.csv    - correlation r, p-value, n per sensor

USAGE
=====
    Run AFTER step4_stl_decomposition.py.
        python step4b_daily_noise_vs_temperature.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds - save
edits as step4b_daily_noise_vs_temperature_v2.py, _v3.py, etc.
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
STEP4_FOLDER = "Step 4 STL Decomposition"
STEP_FOLDER  = "Step 4b Daily Noise vs Temperature"

FONT = "Arial"

SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

TEMP_COL = "Exhibition_Temp(\u00b0C)"
TEMP_COLOUR = "#555555"

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


def daily_range(series_by_date):
    return series_by_date.max() - series_by_date.min()


def main():
    section("Step 4b: Daily Noise vs Temperature (STL components)")

    step1_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP1_FOLDER)
    master_path = safe_path(os.path.join(step1_dir, "All_Sensors_Aligned_Filled_Trimmed.csv"))
    if not os.path.exists(master_path):
        warn(f"Master file not found: {master_path}")
        return
    master = pd.read_csv(master_path, parse_dates=["Timestamp"])
    if TEMP_COL not in master.columns:
        warn(f"Temperature column '{TEMP_COL}' not found in master file. Aborting.")
        return
    ok(f"Loaded master file: {len(master)} rows")

    for start_str, end_str in EXCLUDE_RANGES:
        start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
        mask = (master["Timestamp"] >= start) & (master["Timestamp"] <= end)
        if mask.sum():
            master = master.loc[~mask].copy()
            ok(f"Excluded {mask.sum()} rows in cabling-change window {start} to {end}")

    master["Date"] = master["Timestamp"].dt.date
    daily_temp = master.groupby("Date")[TEMP_COL].apply(daily_range)
    daily_temp.name = "Temp_Daily_Range"

    step4_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP4_FOLDER)
    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    try:
        os.makedirs(safe_path(out_dir), exist_ok=True)
    except PermissionError:
        warn(f"Could not create output folder (permission denied): {out_dir}")
        return

    valid_sensors = []
    for code, (sensor_name, colour) in SENSOR_MAP.items():
        decomp_path = safe_path(os.path.join(step4_dir, f"{sensor_name}_decomposed.csv"))
        if not os.path.exists(decomp_path):
            warn(f"Decomposed file not found for {sensor_name}: {decomp_path} - skipping.")
            continue
        valid_sensors.append((code, sensor_name, colour, decomp_path))

    n_sensors = len(valid_sensors)
    if n_sensors == 0:
        warn("No decomposed files found. Run step4_stl_decomposition.py first.")
        return

    fig = make_subplots(
        rows=n_sensors, cols=1, shared_xaxes=True,
        subplot_titles=[name for _, name, _, _ in valid_sensors],
        specs=[[{"secondary_y": True}]] * n_sensors,
        vertical_spacing=0.03,
    )

    summary_rows = []

    for i, (code, sensor_name, colour, decomp_path) in enumerate(valid_sensors, start=1):
        section(sensor_name)
        decomp = pd.read_csv(decomp_path, parse_dates=["Timestamp"], encoding="utf-8-sig")

        seasonal_col = f"{sensor_name}_Seasonal_Daily(\u00b5S)"
        required_cols = {"Timestamp", seasonal_col}
        if not required_cols.issubset(decomp.columns):
            warn(f"{sensor_name}: expected columns {required_cols} not all found "
                 f"(has {list(decomp.columns)}) - skipping.")
            continue

        for start_str, end_str in EXCLUDE_RANGES:
            start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
            m = (decomp["Timestamp"] >= start) & (decomp["Timestamp"] <= end)
            if m.sum():
                decomp = decomp.loc[~m].copy()

        decomp["Date"] = decomp["Timestamp"].dt.date

        daily_seasonal = decomp.dropna(subset=[seasonal_col]).groupby("Date")[seasonal_col].apply(daily_range)
        daily_seasonal.name = "Seasonal_Daily_Range"

        merged = pd.concat([daily_seasonal, daily_temp], axis=1).dropna()
        if len(merged) < 3:
            warn(f"{sensor_name}: insufficient overlapping daily data - skipped.")
            continue

        r_seasonal, p_seasonal = stats.pearsonr(merged["Temp_Daily_Range"], merged["Seasonal_Daily_Range"])

        ok(f"seasonal_daily vs temp: r={r_seasonal:.3f}  p={p_seasonal:.4g}  n_days={len(merged)}")

        summary_rows.append({
            "Sensor": sensor_name,
            "Seasonal_Daily_r": r_seasonal, "Seasonal_Daily_p": p_seasonal,
            "n_days": len(merged),
        })

        dates = pd.to_datetime(merged.index)

        fig.add_trace(
            go.Scatter(x=dates, y=merged["Seasonal_Daily_Range"], mode="lines",
                       line=dict(color=colour, width=1.5),
                       name=f"{sensor_name} seasonal_daily range", showlegend=False),
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

    html_path = safe_path(os.path.join(out_dir, "daily_seasonal_vs_temperature.html"))
    try:
        fig.write_html(html_path, include_plotlyjs="cdn")
        ok(f"Saved combined plot: {html_path}")
    except PermissionError:
        warn(f"Could not write plot (file open elsewhere?): {html_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = safe_path(os.path.join(out_dir, "daily_seasonal_vs_temperature.csv"))
        try:
            summary_df.to_csv(csv_path, index=False)
            ok(f"Saved correlation summary: {csv_path}")
        except PermissionError:
            warn(f"Could not write summary CSV (file open in Excel?): {csv_path}")

    section("Done")


if __name__ == "__main__":
    main()
