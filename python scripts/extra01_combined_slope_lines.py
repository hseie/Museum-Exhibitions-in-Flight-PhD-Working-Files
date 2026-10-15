"""
extra01_combined_slope_lines.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Extra Analysis (outside the main pipeline)
Plots Step 2's temperature/strain regression line for all six sensors
together on a single graph.

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors:
  1. Loads Step 1's combined master file and takes that sensor's strain
     column together with Exhibition_Temp(°C), dropping rows where either
     is missing.
  2. Fits the same OLS regression Step 2 fits: strain = alpha*temp + beta,
     over the sensor's whole record (no splitting).
  3. Draws that one fitted line, spanning the sensor's own observed
     temperature range, on a single combined plot - each sensor in its
     assigned colour.

OUTPUTS (saved to Output\\Extra Analysis\\Combined Slope Lines\\):
  combined_slope_lines.html   - one graph, all six regression lines
  combined_slope_lines.csv    - alpha, beta, R^2, SE, p-value, n per sensor

USAGE
=====
    Run AFTER step1_collation_alignment.py has produced its master file.
        python extra01_combined_slope_lines.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds - save
edits as extra01_combined_slope_lines_v2.py, _v3.py, etc.
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
ROUND_NAME   = "Round 2"      # which pipeline data-round's Step 1 output to read from
STEP1_FOLDER = "Step 1 Collation Alignment and Raw Plots"
STEP_FOLDER  = "Extra Analysis Steps"

FONT = "Arial"

# Sensor subfolder -> (display name, column name in master file, plot colour)
# Confirmed against actual master file columns via check_columns.py.
SENSOR_MAP = {
    "NP": ("Nose_Portside",    "Nose_Portside_Strain(\u00b5S)",    "#2196F3"),
    "NS": ("Nose_Starboard",   "Nose_Starboard_Strain(\u00b5S)",   "#FF9800"),
    "PB": ("Portside_Bottom",  "Portside_Bottom_Strain(\u00b5S)",  "#4CAF50"),
    "PT": ("Portside_Top",     "Portside_Top_Strain(\u00b5S)",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "Starboard_Bottom_Strain(\u00b5S)", "#F44336"),
    "ST": ("Starboard_Top",    "Starboard_Top_Strain(\u00b5S)",    "#8B4513"),
}

TEMP_COL = "Exhibition_Temp(\u00b0C)"

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
    section("Extra Analysis: Combined Slope Lines")

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

        d = df.dropna(subset=[col_name, TEMP_COL])
        fit = fit_ols(d[TEMP_COL].to_numpy(), d[col_name].to_numpy())
        if fit is None:
            warn(f"{sensor_name}: insufficient/degenerate data - skipped.")
            continue

        x_line = np.array([fit["temp_min"], fit["temp_max"]])
        y_line = fit["alpha"] * x_line + fit["beta"]

        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color=colour, width=2.5),
            name=sensor_name,
        ))

        ok(f"{sensor_name}: alpha={fit['alpha']:.4f}  beta={fit['beta']:.2f}  "
           f"R2={fit['r2']:.3f}  n={fit['n']}")

        summary_rows.append({
            "Sensor": sensor_name, "alpha": fit["alpha"], "beta": fit["beta"],
            "R2": fit["r2"], "SE": fit["se"], "p_value": fit["p_value"], "n": fit["n"],
        })

    fig.update_layout(
        font=dict(family=FONT, size=12),
        xaxis_title=f"Temperature ({TEMP_COL.split('(')[-1].rstrip(')')})",
        yaxis_title="Strain",
        legend=dict(orientation="h", y=-0.25, x=0, xanchor="left",
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#cccccc", borderwidth=1),
        margin=dict(l=65, r=60, t=40, b=110),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)

    html_path = safe_path(os.path.join(out_dir, "combined_slope_lines.html"))
    try:
        fig.write_html(html_path, include_plotlyjs="cdn")
        ok(f"Saved combined plot: {html_path}")
    except PermissionError:
        warn(f"Could not write plot (file open elsewhere?): {html_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = safe_path(os.path.join(out_dir, "combined_slope_lines.csv"))
        try:
            summary_df.to_csv(csv_path, index=False)
            ok(f"Saved summary CSV: {csv_path}")
        except PermissionError:
            warn(f"Could not write summary CSV (file open in Excel?): {csv_path}")

    section("Done")


if __name__ == "__main__":
    main()
