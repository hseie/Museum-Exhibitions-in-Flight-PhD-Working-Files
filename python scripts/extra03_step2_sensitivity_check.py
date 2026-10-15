"""
extra03_step2_sensitivity_check.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Extra Analysis (outside the main pipeline)
Sensitivity check: does excluding days with direct physical interference on
the aircraft/rigging meaningfully change Step 2's temperature regression
(alpha, beta, R²) per sensor?

WHY THIS EXISTS
================
Step 2 fits strain = alpha*temp + beta over the ENTIRE record, with no
filtering for days when people were physically handling the aircraft or its
rigging (which could cause strain excursions unrelated to temperature).
This script re-fits the same regression with those days additionally
excluded, and reports the before/after difference per sensor, so you can
see whether the contamination is large enough to matter before deciding
whether to change the main pipeline.

INTERFERENCE DAYS
==================
Confirmed by the user against the colour-coded event log (yellow = direct
interference with the Canberra specifically, as distinct from events
around other suspended aircraft in the hangar):
  - 2025-08-28 (single day) : Starboard_Top sensor installation
  - 2026-03-16 to 2026-03-17 : cleaning + rigging inspection
  - 2026-05-05 (single day) : rigging cables measured

NOT included (confirmed not to concern the Canberra itself):
  - 2025-09-11 to 2025-09-14 stropping removal (unrelated aircraft)
  - 2026-06-29 rigging change (unrelated aircraft) - the Canberra's own
    rigging cable change is the confirmed 21-23 Jul 2026 cabling window,
    already excluded separately below.

The confirmed cabling-change window (21-23 Jul 2026) is excluded in BOTH
the "before" and "after" fits below, since that one is already established
as a genuine instrumentation artifact, not just a sensitivity question.

WHAT THIS SCRIPT DOES
======================
For each sensor:
  1. Fits the Step 2 regression on all data (cabling window excluded only)
     - this is the "Before" fit, matching what Step 2 currently produces.
  2. Fits the same regression with the interference days ALSO excluded
     - this is the "After" fit.
  3. Reports alpha, beta, R², n, and the % change in alpha/R² between them.

OUTPUTS (saved to Output\\Round 2\\Extra Analysis Steps\\):
  step2_sensitivity_check.csv    - before/after alpha, beta, R², n per sensor
  step2_sensitivity_check.html   - before/after fit lines, per sensor

USAGE
=====
    Run AFTER step1_collation_alignment.py has produced its master file.
        python extra03_step2_sensitivity_check.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds - save
edits as extra03_step2_sensitivity_check_v2.py, _v3.py, etc.
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

OUTPUT_BASE  = "Output"
ROUND_NAME   = "Round 2"
STEP1_FOLDER = "Step 1 Collation Alignment and Raw Plots"
STEP_FOLDER  = "Extra Analysis Steps"

FONT = "Arial"

SENSOR_MAP = {
    "NP": ("Nose_Portside",    "Nose_Portside_Strain(\u00b5S)",    "#2196F3"),
    "NS": ("Nose_Starboard",   "Nose_Starboard_Strain(\u00b5S)",   "#FF9800"),
    "PB": ("Portside_Bottom",  "Portside_Bottom_Strain(\u00b5S)",  "#4CAF50"),
    "PT": ("Portside_Top",     "Portside_Top_Strain(\u00b5S)",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "Starboard_Bottom_Strain(\u00b5S)", "#F44336"),
    "ST": ("Starboard_Top",    "Starboard_Top_Strain(\u00b5S)",    "#8B4513"),
}

TEMP_COL = "Exhibition_Temp(\u00b0C)"

# Confirmed instrumentation artifact - excluded from BOTH fits.
CABLING_CHANGE_RANGE = ("2026-07-21 00:00", "2026-07-23 17:00")

# Physical interference / rigging work - excluded ONLY from the "After" fit.
# Confirmed against the colour-coded event log - see docstring above.
INTERFERENCE_RANGES = [
    ("2025-08-28", "2025-08-28", "Starboard_Top sensor installation"),
    ("2026-03-16", "2026-03-17", "Cleaning + rigging inspection"),
    ("2026-05-05", "2026-05-05", "Rigging cables measured"),
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


def fit_ols(x, y):
    if len(x) < 2 or np.allclose(np.std(x), 0):
        return None
    res = stats.linregress(x, y)
    return {
        "alpha": res.slope, "beta": res.intercept, "r2": res.rvalue ** 2,
        "se": res.stderr, "p_value": res.pvalue, "n": len(x),
        "temp_min": x.min(), "temp_max": x.max(),
    }


def main():
    section("Extra Analysis: Step 2 Sensitivity Check")

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

    # "Before" data: cabling window excluded only (matches current Step 2)
    cab_start, cab_end = pd.Timestamp(CABLING_CHANGE_RANGE[0]), pd.Timestamp(CABLING_CHANGE_RANGE[1])
    cab_mask = (df["Timestamp"] >= cab_start) & (df["Timestamp"] <= cab_end)
    df_before = df.loc[~cab_mask].copy()
    ok(f"Excluded {cab_mask.sum()} rows in confirmed cabling-change window (both fits)")

    # "After" data: also excludes interference days
    interference_mask = pd.Series(False, index=df_before.index)
    for start_str, end_str, desc in INTERFERENCE_RANGES:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        m = (df_before["Timestamp"] >= start) & (df_before["Timestamp"] <= end)
        interference_mask |= m
        ok(f"Interference range '{desc}': {m.sum()} rows ({start_str} to {end_str})")

    df_after = df_before.loc[~interference_mask].copy()
    ok(f"Total interference rows excluded for 'After' fit: {interference_mask.sum()}")

    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    try:
        os.makedirs(safe_path(out_dir), exist_ok=True)
    except PermissionError:
        warn(f"Could not create output folder (permission denied): {out_dir}")
        return

    fig = go.Figure()
    summary_rows = []

    for folder_code, (sensor_name, col_name, colour) in SENSOR_MAP.items():
        if col_name not in df.columns:
            warn(f"Column '{col_name}' not found in master file - skipping.")
            continue

        section(sensor_name)

        d_before = df_before.dropna(subset=[col_name, TEMP_COL])
        fit_before = fit_ols(d_before[TEMP_COL].to_numpy(), d_before[col_name].to_numpy())

        d_after = df_after.dropna(subset=[col_name, TEMP_COL])
        fit_after = fit_ols(d_after[TEMP_COL].to_numpy(), d_after[col_name].to_numpy())

        if fit_before is None or fit_after is None:
            warn(f"{sensor_name}: insufficient data for one of the fits - skipped.")
            continue

        alpha_pct_change = 100 * (fit_after["alpha"] - fit_before["alpha"]) / fit_before["alpha"]
        r2_change = fit_after["r2"] - fit_before["r2"]

        ok(f"Before: alpha={fit_before['alpha']:.4f}  R2={fit_before['r2']:.4f}  n={fit_before['n']}")
        ok(f"After:  alpha={fit_after['alpha']:.4f}  R2={fit_after['r2']:.4f}  n={fit_after['n']}")
        ok(f"Change: alpha {alpha_pct_change:+.2f}%   R2 {r2_change:+.4f}")

        summary_rows.append({
            "Sensor": sensor_name,
            "Alpha_Before": fit_before["alpha"], "Alpha_After": fit_after["alpha"],
            "Alpha_Pct_Change": alpha_pct_change,
            "Beta_Before": fit_before["beta"], "Beta_After": fit_after["beta"],
            "R2_Before": fit_before["r2"], "R2_After": fit_after["r2"],
            "R2_Change": r2_change,
            "N_Before": fit_before["n"], "N_After": fit_after["n"],
            "N_Excluded": fit_before["n"] - fit_after["n"],
        })

        for fit, dash, label in [(fit_before, "solid", "Before"), (fit_after, "dot", "After")]:
            x_line = np.array([fit["temp_min"], fit["temp_max"]])
            y_line = fit["alpha"] * x_line + fit["beta"]
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color=colour, width=2.5, dash=dash),
                name=f"{sensor_name} — {label}",
            ))

    fig.update_layout(
        font=dict(family=FONT, size=12),
        xaxis_title=f"Temperature ({TEMP_COL.split('(')[-1].rstrip(')')})",
        yaxis_title="Strain",
        legend=dict(orientation="h", y=-0.3, x=0, xanchor="left",
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#cccccc", borderwidth=1),
        margin=dict(l=65, r=60, t=40, b=140),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)

    html_path = safe_path(os.path.join(out_dir, "step2_sensitivity_check.html"))
    try:
        fig.write_html(html_path, include_plotlyjs="cdn")
        ok(f"Saved before/after plot: {html_path}")
    except PermissionError:
        warn(f"Could not write plot (file open elsewhere?): {html_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = safe_path(os.path.join(out_dir, "step2_sensitivity_check.csv"))
        try:
            summary_df.to_csv(csv_path, index=False)
            ok(f"Saved sensitivity comparison: {csv_path}")
        except PermissionError:
            warn(f"Could not write summary CSV (file open in Excel?): {csv_path}")

    section("Done")


if __name__ == "__main__":
    main()
