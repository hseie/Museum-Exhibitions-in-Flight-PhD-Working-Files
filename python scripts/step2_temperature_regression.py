"""
step2_temperature_regression.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 2
Linear (OLS) regression of each sensor's strain against exhibition
temperature, on the 15-minute aligned, gap-filled, settling-trimmed data
produced by Step 1.

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors:
  1. Loads Step 1's combined master file (All_Sensors_Aligned_Filled_
     Trimmed.csv) and takes that sensor's strain column together with
     Exhibition_Temp(°C), dropping rows where either is missing.
  2. Fits an OLS regression: strain = alpha * temperature + beta.
  3. Plots a scatter of strain vs temperature, points coloured by days
     since that sensor's own monitoring start (so drift/seasonal shift
     is visible), with the OLS fit line overlaid and the equation, R²,
     and n annotated directly on the plot.
  4. Records alpha, beta, R², standard error, p-value, and n per sensor
     to a summary CSV — this is what Step 3 (temperature compensation)
     will read alpha from.

OUTPUTS (saved to Output\\Round 1\\Step 2 Temperature Regression\\):
  regression_<Sensor>.html   - scatter + OLS fit, per sensor (x6)
  regression_summary.csv     - alpha, beta, R², SE, p-value, n per sensor

USAGE
=====
    Run AFTER step1_collation_alignment.py has produced its output.
        python step2_temperature_regression.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds — save
edits as step2_temperature_regression_v2.py, _v3.py, etc.
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
ROUND_NAME   = "Round 1"
STEP1_FOLDER = "Step 1 Collation Alignment and Raw Plots"
STEP_FOLDER  = "Step 2 Temperature Regression"

FONT = "Arial"

# Sensor subfolder -> (internal key used in column names, plot colour)
# Must match Step 1 exactly.
SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

TEMP_COL = "Exhibition_Temp(\u00b0C)"

# ══════════════════════════════════════════════════════════════════════════
# END USER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════


# ── Small helpers ────────────────────────────────────────────────────────────

def section(msg):
    print(f"\n{'-'*70}\n  {msg}\n{'-'*70}")


def ok(msg):
    print(f"  OK  {msg}")


def warn(msg):
    print(f"  !!  {msg}")


def safe_path(path):
    """
    Windows has a classic 260-character MAX_PATH limit, which a deep
    OneDrive folder structure can exceed. Prepending the '\\\\?\\'
    extended-length prefix bypasses that limit at the Win32 API level for
    absolute paths — no admin rights or system changes needed. No effect
    on non-Windows systems.
    """
    if os.name == "nt":
        abs_path = os.path.abspath(path)
        if not abs_path.startswith("\\\\?\\"):
            abs_path = "\\\\?\\" + abs_path
        return abs_path
    return path


def get_step1_dir():
    return os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP1_FOLDER)


def get_output_dir():
    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    os.makedirs(safe_path(out_dir), exist_ok=True)
    return out_dir


def save_html(fig, path):
    fig.write_html(safe_path(path), include_plotlyjs="cdn")
    ok(f"Saved: {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    step1_dir = get_step1_dir()
    out_dir   = get_output_dir()
    print(f"Reading Step 1 output from: {step1_dir}")
    print(f"Output folder: {out_dir}")

    master_path = os.path.join(step1_dir, "All_Sensors_Aligned_Filled_Trimmed.csv")
    if not os.path.isfile(safe_path(master_path)):
        raise SystemExit(
            f"Could not find Step 1's combined master file:\n  {master_path}\n"
            f"Run step1_collation_alignment.py first."
        )
    master = pd.read_csv(safe_path(master_path), parse_dates=["Timestamp"])
    ok(f"Loaded master file: {len(master):,} rows")

    if TEMP_COL not in master.columns:
        raise SystemExit(f"Expected temperature column '{TEMP_COL}' not found in master file.")

    section("STEP 2 - Temperature regression (OLS, per sensor)")

    summary_rows = []

    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        label = sensor_key.replace("_", " ")

        if strain_col not in master.columns:
            warn(f"{sensor_key}: column not found in master file — skipped")
            continue

        mask = master[strain_col].notna() & master[TEMP_COL].notna()
        n = int(mask.sum())
        if n < 30:
            warn(f"{sensor_key}: only {n} valid rows — insufficient for regression, skipped")
            continue

        T = master.loc[mask, TEMP_COL].to_numpy()
        S = master.loc[mask, strain_col].to_numpy()
        dates = master.loc[mask, "Timestamp"]

        slope, intercept, r_val, p_val, _std_err = stats.linregress(T, S)
        r2 = r_val ** 2
        S_hat = slope * T + intercept
        residual_se = float(np.sqrt(np.sum((S - S_hat) ** 2) / (n - 2)))

        days_since_start = (dates - dates.min()).dt.total_seconds() / 86400.0

        summary_rows.append({
            "Sensor": label,
            "alpha (\u00b5S/\u00b0C)": round(slope, 4),
            "beta (\u00b5S)": round(intercept, 4),
            "R2": round(r2, 4),
            "SE": round(residual_se, 4),
            "p-value": p_val,
            "N": n,
        })
        ok(f"{sensor_key}: alpha={slope:.4f}  beta={intercept:.4f}  "
           f"R2={r2:.4f}  n={n:,}")

        # ── Scatter plot: strain vs temperature, coloured by days since start ──
        T_range = np.linspace(T.min(), T.max(), 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=T, y=S, mode="markers",
            marker=dict(
                color=days_since_start, colorscale="Viridis", size=3,
                opacity=0.45,
                colorbar=dict(
                    title=dict(text="Days since monitoring start", side="right"),
                    thickness=14, len=0.75, y=0.55, yanchor="middle",
                ),
                showscale=True,
            ),
            name="Observed",
            hovertemplate="T=%{x:.2f} \u00b0C<br>\u03b5=%{y:.3f} \u00b5S<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=T_range, y=slope * T_range + intercept,
            mode="lines", name="OLS fit",
            line=dict(color=colour, width=2.5),
        ))

        eq_text = (
            f"\u03b5 = {slope:.4f} \u00b7 T + {intercept:.4f}<br>"
            f"R\u00b2 = {r2:.4f}     n = {n:,}"
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.97,
            text=eq_text, showarrow=False,
            font=dict(size=13, family=FONT, color="#222222"),
            align="left", bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#bbbbbb", borderwidth=1, borderpad=6,
            xanchor="left", yanchor="top",
        )

        fig.update_layout(
            title=dict(
                text=f"{label} - Strain vs Exhibition Temperature (OLS regression)",
                font=dict(size=14, family=FONT), x=0.0, xanchor="left",
            ),
            xaxis=dict(title="Exhibition Temperature (\u00b0C)",
                       showgrid=True, gridcolor="#eeeeee"),
            yaxis=dict(title="Strain (\u00b5S)",
                       showgrid=True, gridcolor="#eeeeee"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=640,
            legend=dict(orientation="h", y=-0.14, x=0, xanchor="left",
                        font=dict(size=11), bgcolor="rgba(255,255,255,0.95)",
                        bordercolor="#cccccc", borderwidth=1),
            margin=dict(l=70, r=110, t=70, b=100),
            font=dict(family=FONT),
        )
        save_html(fig, os.path.join(out_dir, f"regression_{sensor_key}.html"))

    if not summary_rows:
        raise SystemExit("No sensor had enough data for regression — nothing to save.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        safe_path(os.path.join(out_dir, "regression_summary.csv")),
        index=False, encoding="utf-8-sig",
    )
    ok("regression_summary.csv saved")

    section("STEP 2 COMPLETE")
    print(f"  All outputs saved to:\n  {out_dir}")


if __name__ == "__main__":
    main()
