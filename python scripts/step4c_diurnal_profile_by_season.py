"""
step4c_diurnal_profile_by_season.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 4c
Shows the typical shape of each sensor's daily cycle - which time of day
tends to read high vs low - split by season, so you can see not just how
BIG the daily swing is (Step 4b) but WHEN within the day it happens, and
whether that shape changes across seasons.

WHERE THIS SITS
=================
Runs AFTER Step 4 (STL Decomposition), using the same per-sensor decomposed
files as Step 4b. Placed as "4c" - a standalone diagnostic, nothing later
in the pipeline depends on its output.

WHY seasonal_daily, NOT raw/compensated strain
=================================================
The seasonal_daily component already isolates the systematic 24-hour cycle
from trend and irregular noise, so averaging it by time-of-day gives a much
cleaner diurnal shape than averaging the noisier compensated signal would.
This matches Step 4b's approach.

SEASON DEFINITION
===================
Standard meteorological seasons, by calendar month (year ignored, since the
record spans parts of two years for most months):
  Winter: Dec, Jan, Feb   Spring: Mar, Apr, May
  Summer: Jun, Jul, Aug   Autumn: Sep, Oct, Nov

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors:
  1. Loads Step 4's per-sensor decomposed CSV.
  2. Excludes the confirmed cabling-change window (21-23 Jul 2026).
  3. Assigns each 15-minute reading a season (by month) and a time-of-day
     (HH:MM, ignoring the date).
  4. Averages seasonal_daily across all days within each (season,
     time-of-day) group - producing one typical 24-hour curve per season.
  5. Plots all four seasonal curves together per sensor, so peaks/troughs
     and their timing are directly comparable.
  6. Reports, per sensor per season, the time-of-day of the average peak
     and trough, and the peak-to-trough amplitude - i.e. whether nights or
     days read higher, and by how much, per season.

OUTPUTS (saved to Output\\Round 2\\Step 4c Diurnal Profile by Season\\):
  diurnal_profile_by_season.html   - one panel per sensor, 4 seasonal curves
  diurnal_profile_by_season.csv    - peak/trough time & value per sensor/season

USAGE
=====
    Run AFTER step4_stl_decomposition.py.
        python step4c_diurnal_profile_by_season.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds - save
edits as step4c_diurnal_profile_by_season_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import warnings

import numpy as np
import pandas as pd
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
STEP4_FOLDER = "Step 4 STL Decomposition"
STEP_FOLDER  = "Step 4c Diurnal Profile by Season"

FONT = "Arial"

SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

EXCLUDE_RANGES = [
    ("2026-07-21 00:00", "2026-07-23 17:00"),
]

MONTH_TO_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

SEASON_COLOURS = {
    "Winter": "#2196F3",
    "Spring": "#4CAF50",
    "Summer": "#FF9800",
    "Autumn": "#8B4513",
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

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
    section("Step 4c: Diurnal Profile by Season")

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
        rows=n_sensors, cols=1,
        subplot_titles=[name for _, name, _, _ in valid_sensors],
        vertical_spacing=0.05,
    )

    summary_rows = []

    for i, (code, sensor_name, colour, decomp_path) in enumerate(valid_sensors, start=1):
        section(sensor_name)
        decomp = pd.read_csv(decomp_path, parse_dates=["Timestamp"], encoding="utf-8-sig")

        seasonal_col = f"{sensor_name}_Seasonal_Daily(\u00b5S)"
        if not {"Timestamp", seasonal_col}.issubset(decomp.columns):
            warn(f"{sensor_name}: expected column '{seasonal_col}' not found "
                 f"(has {list(decomp.columns)}) - skipping.")
            continue

        for start_str, end_str in EXCLUDE_RANGES:
            start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
            m = (decomp["Timestamp"] >= start) & (decomp["Timestamp"] <= end)
            if m.sum():
                decomp = decomp.loc[~m].copy()

        decomp = decomp.dropna(subset=[seasonal_col])
        decomp["Season"] = decomp["Timestamp"].dt.month.map(MONTH_TO_SEASON)
        decomp["TimeOfDay"] = decomp["Timestamp"].dt.strftime("%H:%M")
        decomp["MinutesOfDay"] = decomp["Timestamp"].dt.hour * 60 + decomp["Timestamp"].dt.minute

        for season in SEASON_ORDER:
            season_data = decomp.loc[decomp["Season"] == season]
            if season_data.empty:
                warn(f"{sensor_name} / {season}: no data - skipped.")
                continue

            profile = season_data.groupby(["MinutesOfDay", "TimeOfDay"])[seasonal_col].mean().reset_index()
            profile = profile.sort_values("MinutesOfDay")

            n_days = season_data["Timestamp"].dt.date.nunique()
            ok(f"{season}: n_days={n_days}, n_readings={len(season_data)}")

            fig.add_trace(
                go.Scatter(
                    x=profile["TimeOfDay"], y=profile[seasonal_col], mode="lines",
                    line=dict(color=SEASON_COLOURS[season], width=2),
                    name=season, showlegend=(i == 1),
                    legendgroup=season,
                ),
                row=i, col=1,
            )

            peak_idx = profile[seasonal_col].idxmax()
            trough_idx = profile[seasonal_col].idxmin()
            peak_time = profile.loc[peak_idx, "TimeOfDay"]
            peak_val = profile.loc[peak_idx, seasonal_col]
            trough_time = profile.loc[trough_idx, "TimeOfDay"]
            trough_val = profile.loc[trough_idx, seasonal_col]

            summary_rows.append({
                "Sensor": sensor_name, "Season": season,
                "Peak_Time": peak_time, "Peak_Value": peak_val,
                "Trough_Time": trough_time, "Trough_Value": trough_val,
                "Amplitude": peak_val - trough_val,
                "N_Days": n_days,
            })

        fig.update_yaxes(title_text="Strain (seasonal daily)", row=i, col=1)
        if i == n_sensors:
            fig.update_xaxes(title_text="Time of day", row=i, col=1,
                              tickmode="array",
                              tickvals=["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"])
        else:
            fig.update_xaxes(showticklabels=False, row=i, col=1)

    fig.update_layout(
        font=dict(family=FONT, size=11),
        height=280 * n_sensors,
        legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
        margin=dict(l=65, r=40, t=60, b=60),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")

    html_path = safe_path(os.path.join(out_dir, "diurnal_profile_by_season.html"))
    try:
        fig.write_html(html_path, include_plotlyjs="cdn")
        ok(f"Saved combined plot: {html_path}")
    except PermissionError:
        warn(f"Could not write plot (file open elsewhere?): {html_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = safe_path(os.path.join(out_dir, "diurnal_profile_by_season.csv"))
        try:
            summary_df.to_csv(csv_path, index=False)
            ok(f"Saved peak/trough summary: {csv_path}")
        except PermissionError:
            warn(f"Could not write summary CSV (file open in Excel?): {csv_path}")

    section("Done")


if __name__ == "__main__":
    main()
