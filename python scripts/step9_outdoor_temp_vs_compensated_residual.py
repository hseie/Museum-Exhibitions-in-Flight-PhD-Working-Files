"""
step9_outdoor_temp_vs_compensated_residual.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 9 (standalone, NOT part of run_full_pipeline.py)
Tests whether outdoor temperature explains variance in strain that is left
over AFTER Step 3's indoor-temperature compensation - the direct test of
the building-flexing hypothesis raised with supervisors: does outdoor
temperature affect the aircraft via the building/rigging, independently of
the aircraft's own response to indoor air temperature?

WHY Compensated STRAIN, NOT RAW STRAIN
=========================================
Step 3's compensated strain = raw strain minus the fitted indoor-temperature
relationship (alpha * (T_indoor - T0)). By construction, this residual has
already had the indoor-temperature effect removed - so if outdoor
temperature STILL correlates with it, that correlation cannot be coming
from indoor temperature (it's already been subtracted out). It has to be
coming from something else - which is exactly what the building-flexing
hypothesis predicts.

BUILDS ON STEP 8
==================
Step 8 established the building itself is damped (~25% of outdoor's daily
swing) and lagged (~2h) relative to outdoor. This script re-uses that same
lag-scanning approach, but against compensated strain instead of indoor
temperature, and per sensor rather than as one building-wide number - if
the building-flexing pathway reaches the aircraft via the rigging, it may
not affect all six sensors equally.

NOT part of the main pipeline
===============================
Standalone, one-off analysis. Deliberately NOT wired into
run_full_pipeline.py.

CAUTION ON THE HOUR-LEVEL LAG SCAN
=====================================
The 0-6h lag scan (on raw 15-minute values) can be confounded by shared
daily periodicity: compensated strain still carries its own diurnal cycle
(compensation only removed the linear INDOOR-temperature relationship, not
the whole daily pattern), and outdoor temperature cycles daily too. Shifting
one against the other by a few hours changes how their ~24h cycles line up
in phase, which can look like a "lag effect" even with no real causal delay.
A genuine physical lag should show the correlation curve rising to a peak
WITHIN the 0-6h window and falling away on both sides; a curve that just
slides monotonically to the edge is more likely a phase-alignment artifact
than a real delay - treat those results with caution.

DAY-LEVEL LAG SCAN (added to address the above)
===================================================
To sidestep the phase-alignment problem, this script ALSO runs a separate
lag scan on the DAILY RANGE series (0-5 whole days) - i.e. does outdoor's
swing size N days ago predict today's compensated-strain swing size? Because
this compares day-level aggregates shifted by whole days, it has no sub-day
periodicity to be confounded by, and is the more trustworthy of the two lag
tests here.

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors:
  1. Loads Step 3's per-sensor compensated strain.
  2. Loads the outdoor weather station CSV (same parsing as Step 8).
  3. Excludes the confirmed cabling-change window (21-23 Jul 2026).
  4. Aligns compensated strain and outdoor temperature (8-minute tolerance).
  5. Scans lags 0-6h (raw 15-minute values) between outdoor temperature and
     compensated strain - see the caution above before trusting this one.
  6. Computes daily RANGE of compensated strain and of outdoor temperature,
     then scans lags 0-5 DAYS between the two daily-range series - the more
     trustworthy timing test.
  7. Plots compensated strain against outdoor temperature (dual-axis), the
     hour-lag correlation curve, and the day-lag correlation curve, per
     sensor.

OUTPUTS (saved to Output\\Round 2\\Step 9 Outdoor Temp vs Compensated Residual\\):
  <Sensor>_compensated_vs_outdoor.html   - one file per sensor: time series
                                            overlay + both lag-correlation
                                            curves (hour-level and day-level)
  outdoor_vs_compensated_summary.csv     - hour-lag results, day-lag results,
                                            and same-day daily-range r, per
                                            sensor

USAGE
=====
    Run AFTER step3_temperature_compensation.py (or run_full_pipeline.py,
    which produces the same Step 3 output).
        python step9_outdoor_temp_vs_compensated_residual.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds - save
edits as step9_outdoor_temp_vs_compensated_residual_v2.py, _v3.py, etc.
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
STEP3_FOLDER = "Step 3 Compensation"
STEP_FOLDER  = "Step 9 Outdoor Temp vs Compensated Residual"

OUTSIDE_TEMP_DIR  = "Outside Enviornment"   # matches actual folder name (as given)
OUTSIDE_TEMP_FILE = "Outside Temp.csv"      # confirmed via `dir` - space, not underscore

FONT = "Arial"
OUTSIDE_VALUE_COL = "Value (\u00b0C)"
OUTSIDE_COLOUR = "#FF9800"

SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

SNAP_TOLERANCE_MIN = 8   # matches project convention for near-grid timestamp matching
LAG_HOURS_RANGE = range(0, 7)
DAY_LAG_RANGE = range(0, 6)   # 0-5 whole days, on the daily-range series

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


def parse_outside_timestamp(series):
    """Strip trailing 'BST'/'GMT' suffix and parse as naive local time."""
    stripped = series.str.replace(r"\s+(BST|GMT)$", "", regex=True)
    return pd.to_datetime(stripped, format="%d-%b-%y %I:%M:%S %p", errors="coerce")


def exclude_ranges(timestamp_series, ranges):
    keep = pd.Series(True, index=timestamp_series.index)
    for start_str, end_str in ranges:
        start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
        keep &= ~((timestamp_series >= start) & (timestamp_series <= end))
    return keep


def daily_range(series_by_date):
    return series_by_date.max() - series_by_date.min()


def load_outside_temp():
    outside_path = safe_path(os.path.join(BASE_DIR, OUTSIDE_TEMP_DIR, OUTSIDE_TEMP_FILE))
    if not os.path.exists(outside_path):
        warn(f"Outdoor temperature file not found: {outside_path}")
        return None

    raw = pd.read_csv(outside_path, encoding="utf-8-sig")
    if "Timestamp" not in raw.columns or OUTSIDE_VALUE_COL not in raw.columns:
        warn(f"Expected columns 'Timestamp' and '{OUTSIDE_VALUE_COL}' not found "
             f"(has {list(raw.columns)}).")
        return None

    raw["Timestamp"] = parse_outside_timestamp(raw["Timestamp"])
    n_unparsed = raw["Timestamp"].isna().sum()
    if n_unparsed:
        warn(f"{n_unparsed} outdoor timestamps failed to parse and were dropped.")
    outside = raw.dropna(subset=["Timestamp", OUTSIDE_VALUE_COL]).copy()
    outside = outside.sort_values("Timestamp").reset_index(drop=True)
    outside = outside.rename(columns={OUTSIDE_VALUE_COL: "Outside_Temp(\u00b0C)"})
    ok(f"Loaded outdoor temperature: {len(outside)} rows "
       f"({outside['Timestamp'].min()} to {outside['Timestamp'].max()})")
    return outside


def main():
    section("Step 9: Outdoor Temperature vs Compensated Strain Residual")

    outside = load_outside_temp()
    if outside is None:
        return
    keep = exclude_ranges(outside["Timestamp"], EXCLUDE_RANGES)
    if (~keep).sum():
        ok(f"Excluded {(~keep).sum()} outdoor rows in confirmed cabling-change window")
    outside = outside.loc[keep].reset_index(drop=True)

    step3_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP3_FOLDER)
    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    try:
        os.makedirs(safe_path(out_dir), exist_ok=True)
    except PermissionError:
        warn(f"Could not create output folder (permission denied): {out_dir}")
        return

    summary_rows = []

    for code, (sensor_name, colour) in SENSOR_MAP.items():
        section(sensor_name)
        comp_path = safe_path(os.path.join(step3_dir, f"{sensor_name}_compensated.csv"))
        if not os.path.exists(comp_path):
            warn(f"Compensated file not found for {sensor_name}: {comp_path} - skipping.")
            continue

        comp = pd.read_csv(comp_path, parse_dates=["Timestamp"], encoding="utf-8-sig")
        comp_col = f"{sensor_name}_Compensated(\u00b5S)"
        if comp_col not in comp.columns:
            warn(f"{sensor_name}: expected column '{comp_col}' not found "
                 f"(has {list(comp.columns)}) - skipping.")
            continue

        comp = comp[["Timestamp", comp_col]].dropna()
        keep_c = exclude_ranges(comp["Timestamp"], EXCLUDE_RANGES)
        if (~keep_c).sum():
            ok(f"Excluded {(~keep_c).sum()} compensated-strain rows in cabling-change window")
        comp = comp.loc[keep_c].sort_values("Timestamp").reset_index(drop=True)

        merged = pd.merge_asof(
            comp, outside, on="Timestamp", direction="nearest",
            tolerance=pd.Timedelta(minutes=SNAP_TOLERANCE_MIN),
        ).dropna(subset=["Outside_Temp(\u00b0C)"])

        if len(merged) < 30:
            warn(f"{sensor_name}: too few aligned rows ({len(merged)}) - skipping.")
            continue
        ok(f"Aligned {len(merged)} rows within {SNAP_TOLERANCE_MIN}-minute tolerance")

        # ── Lag scan ─────────────────────────────────────────────────────
        ts_indexed = merged.set_index("Timestamp")
        lag_results = []
        for lag_hours in LAG_HOURS_RANGE:
            shifted = ts_indexed["Outside_Temp(\u00b0C)"].shift(freq=pd.Timedelta(hours=lag_hours))
            aligned = pd.concat([ts_indexed[comp_col], shifted], axis=1).dropna()
            aligned.columns = ["comp", "outside_shifted"]
            if len(aligned) < 30:
                continue
            r, p = stats.pearsonr(aligned["outside_shifted"], aligned["comp"])
            lag_results.append({"Lag_Hours": lag_hours, "r": r, "p_value": p, "n": len(aligned)})

        if not lag_results:
            warn(f"{sensor_name}: insufficient data for lag scan - skipping.")
            continue

        lag_df = pd.DataFrame(lag_results)
        best_row = lag_df.loc[lag_df["r"].abs().idxmax()]
        lag0_row = lag_df.loc[lag_df["Lag_Hours"] == 0].iloc[0]
        ok(f"r at lag 0h: {lag0_row['r']:.4f}   Best: lag {int(best_row['Lag_Hours'])}h, "
           f"r={best_row['r']:.4f} (p={best_row['p_value']:.4g})")

        # ── Daily range comparison (same-day) ───────────────────────────
        merged["Date"] = merged["Timestamp"].dt.date
        daily_comp = merged.groupby("Date")[comp_col].agg(daily_range)
        daily_outside = merged.groupby("Date")["Outside_Temp(\u00b0C)"].agg(daily_range)
        daily = pd.concat([daily_comp, daily_outside], axis=1).dropna()
        daily.columns = ["Comp_Daily_Range", "Outside_Daily_Range"]
        if len(daily) >= 3:
            r_range, p_range = stats.pearsonr(daily["Outside_Daily_Range"], daily["Comp_Daily_Range"])
            ok(f"Daily range correlation (same day): r={r_range:.4f}  p={p_range:.4g}  n_days={len(daily)}")
        else:
            r_range, p_range = np.nan, np.nan

        # ── Day-level lag scan on the daily-range series (avoids the ────
        # ── sub-day periodicity confound the hour-lag scan is exposed to)
        daily_indexed = daily.copy()
        daily_indexed.index = pd.to_datetime(daily_indexed.index)
        day_lag_results = []
        for day_lag in DAY_LAG_RANGE:
            shifted_outside = daily_indexed["Outside_Daily_Range"].shift(day_lag)
            aligned = pd.concat([daily_indexed["Comp_Daily_Range"], shifted_outside], axis=1).dropna()
            aligned.columns = ["comp", "outside_shifted"]
            if len(aligned) < 10:
                continue
            r, p = stats.pearsonr(aligned["outside_shifted"], aligned["comp"])
            day_lag_results.append({"Day_Lag": day_lag, "r": r, "p_value": p, "n": len(aligned)})

        if day_lag_results:
            day_lag_df = pd.DataFrame(day_lag_results)
            best_day_row = day_lag_df.loc[day_lag_df["r"].idxmax()]
            ok(f"Day-lag scan: best day-lag={int(best_day_row['Day_Lag'])}  "
               f"r={best_day_row['r']:.4f}  p={best_day_row['p_value']:.4g}")
        else:
            day_lag_df = pd.DataFrame(columns=["Day_Lag", "r", "p_value", "n"])
            best_day_row = None
            warn(f"{sensor_name}: insufficient data for day-lag scan.")

        summary_rows.append({
            "Sensor": sensor_name,
            "r_at_lag0": lag0_row["r"], "p_at_lag0": lag0_row["p_value"],
            "Best_Lag_Hours": int(best_row["Lag_Hours"]), "Best_Lag_r": best_row["r"],
            "Best_Lag_p": best_row["p_value"],
            "Daily_Range_r_sameday": r_range, "Daily_Range_p_sameday": p_range,
            "Best_Day_Lag": int(best_day_row["Day_Lag"]) if best_day_row is not None else np.nan,
            "Best_Day_Lag_r": best_day_row["r"] if best_day_row is not None else np.nan,
            "Best_Day_Lag_p": best_day_row["p_value"] if best_day_row is not None else np.nan,
            "n_rows": len(merged),
        })

        # ── Plot: time series overlay + hour-lag curve + day-lag curve ──
        fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25],
                            specs=[[{"secondary_y": True}], [{}], [{}]],
                            subplot_titles=[f"{sensor_name}: Compensated strain vs Outdoor temperature",
                                            "Correlation vs outdoor lag (hours, raw values - see caution in docstring)",
                                            "Correlation vs outdoor lag (days, daily-range series)"],
                            vertical_spacing=0.10)
        fig.add_trace(go.Scatter(x=merged["Timestamp"], y=merged[comp_col], mode="lines",
                                  line=dict(color=colour, width=1), name="Compensated strain"),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=merged["Timestamp"], y=merged["Outside_Temp(\u00b0C)"], mode="lines",
                                  line=dict(color=OUTSIDE_COLOUR, width=1), name="Outdoor temp"),
                      row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Strain (\u00b5S)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Outdoor \u00b0C", row=1, col=1, secondary_y=True)

        fig.add_trace(go.Scatter(x=lag_df["Lag_Hours"], y=lag_df["r"], mode="lines+markers",
                                  line=dict(color=colour, width=2), showlegend=False),
                      row=2, col=1)
        fig.update_xaxes(title_text="Outdoor lag (hours)", row=2, col=1)
        fig.update_yaxes(title_text="r", row=2, col=1)

        if not day_lag_df.empty:
            fig.add_trace(go.Scatter(x=day_lag_df["Day_Lag"], y=day_lag_df["r"], mode="lines+markers",
                                      line=dict(color=colour, width=2, dash="dot"), showlegend=False),
                          row=3, col=1)
        fig.update_xaxes(title_text="Outdoor lag (days)", row=3, col=1)
        fig.update_yaxes(title_text="r", row=3, col=1)

        fig.update_layout(
            font=dict(family=FONT, size=11), height=850,
            legend=dict(orientation="h", y=1.04, x=0, xanchor="left"),
            margin=dict(l=65, r=60, t=70, b=50), plot_bgcolor="white",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
        fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")

        html_path = safe_path(os.path.join(out_dir, f"{sensor_name}_compensated_vs_outdoor.html"))
        try:
            fig.write_html(html_path, include_plotlyjs="cdn")
            ok(f"Saved: {os.path.basename(html_path)}")
        except PermissionError:
            warn(f"Could not write plot (file open elsewhere?): {html_path}")

    if summary_rows:
        csv_path = safe_path(os.path.join(out_dir, "outdoor_vs_compensated_summary.csv"))
        try:
            pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
            ok(f"Saved: {os.path.basename(csv_path)}")
        except PermissionError:
            warn(f"Could not write summary CSV (file open in Excel?): {csv_path}")
    else:
        warn("No sensors produced results - summary CSV not written.")

    section("Done")


if __name__ == "__main__":
    main()
