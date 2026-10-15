"""
step10_indoor_outdoor_multiple_regression.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 10 (standalone, NOT part of run_full_pipeline.py)
Direct test of the building-flexing hypothesis: does outdoor temperature
explain strain variance that indoor temperature does NOT already explain?

WHY A MULTIPLE REGRESSION, NOT TWO SEPARATE SINGLE-VARIABLE MODELS
======================================================================
Indoor and outdoor temperature are themselves correlated (Step 8 found
same-day r=0.84), so fitting strain against either one ALONE would let
that one variable soak up variance genuinely coming from both pathways -
it can't tell you which pathway the shared variance actually belongs to.

Fitting BOTH together in one model (strain = a1*indoor + a2*outdoor + b)
solves this: each coefficient reflects only the variance uniquely
attributable to that variable, after accounting for the other. This
directly answers "does outdoor still matter once indoor is already
accounted for" - the building-flexing question.

NOT part of the main pipeline, and does NOT change Step 3's compensation
=============================================================================
This is a standalone diagnostic. It does not modify, replace, or feed back
into Step 2/3's existing indoor-only linear compensation - it fits its own
separate model on RAW strain, purely to test whether outdoor adds anything.
Nothing here is "adopted" into the pipeline; it's read-only diagnostic
information.

MULTICOLLINEARITY CHECK
==========================
Because indoor and outdoor are correlated, this script also reports the
Variance Inflation Factor (VIF) for both predictors. A VIF well above ~5
would mean the two variables are too entangled to reliably separate their
individual effects - worth checking before trusting the individual
coefficients (the model can still be valid overall even with moderate VIF,
but individual coefficient estimates become less stable as it rises).

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors:
  1. Loads Step 1's raw strain + indoor temperature.
  2. Loads the outdoor weather station CSV (same parsing as Step 8/9).
  3. Excludes the confirmed cabling-change window (21-23 Jul 2026).
  4. Aligns all three series (8-minute tolerance).
  5. Fits Model A: strain = a*indoor + b               (indoor-only, OLS)
  6. Fits Model B: strain = a1*indoor + a2*outdoor + b  (both together, OLS)
  7. Reports R^2 for both models, the R^2 gain from adding outdoor, an
     F-test on whether that gain is statistically significant, both
     raw and standardized coefficients for indoor/outdoor in Model B, and
     VIF for both predictors.
  8. Plots an "added-variable" (partial regression) plot per sensor: strain
     residual after removing indoor's effect, against outdoor residual
     after removing indoor's effect - the standard way to visualise a
     second predictor's marginal contribution.

OUTPUTS (saved to Output\\Round 2\\Step 10 Indoor Outdoor Multiple Regression\\):
  <Sensor>_added_variable_plot.html   - one file per sensor
  multiple_regression_summary.csv     - R^2 (both models), R^2 gain, F-test,
                                         standardized coefficients, VIF, per
                                         sensor

USAGE
=====
    Run AFTER step1_collation_alignment.py has produced its master file.
        python step10_indoor_outdoor_multiple_regression.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds - save
edits as step10_indoor_outdoor_multiple_regression_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
STEP_FOLDER   = "Step 10 Indoor Outdoor Multiple Regression"

OUTSIDE_TEMP_DIR  = "Outside Enviornment"   # matches actual folder name (as given)
OUTSIDE_TEMP_FILE = "Outside Temp.csv"      # confirmed via `dir` - space, not underscore

FONT = "Arial"
INDOOR_TEMP_COL  = "Exhibition_Temp(\u00b0C)"
OUTSIDE_VALUE_COL = "Value (\u00b0C)"

SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

SNAP_TOLERANCE_MIN = 8

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
    if os.name == "nt":
        abspath = os.path.abspath(path)
        if not abspath.startswith("\\\\?\\"):
            return "\\\\?\\" + abspath
    return path


def parse_outside_timestamp(series):
    stripped = series.str.replace(r"\s+(BST|GMT)$", "", regex=True)
    return pd.to_datetime(stripped, format="%d-%b-%y %I:%M:%S %p", errors="coerce")


def exclude_ranges(timestamp_series, ranges):
    keep = pd.Series(True, index=timestamp_series.index)
    for start_str, end_str in ranges:
        start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
        keep &= ~((timestamp_series >= start) & (timestamp_series <= end))
    return keep


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
    section("Step 10: Indoor + Outdoor Multiple Regression")

    step1_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP1_FOLDER)
    master_path = safe_path(os.path.join(step1_dir, "All_Sensors_Aligned_Filled_Trimmed.csv"))
    if not os.path.exists(master_path):
        warn(f"Master file not found: {master_path}")
        warn("Run step1_collation_alignment.py first, or check ROUND_NAME.")
        return
    master = pd.read_csv(master_path, parse_dates=["Timestamp"])
    if INDOOR_TEMP_COL not in master.columns:
        warn(f"Indoor temperature column '{INDOOR_TEMP_COL}' not found. Aborting.")
        return
    ok(f"Loaded indoor master file: {len(master)} rows")

    outside = load_outside_temp()
    if outside is None:
        return

    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    try:
        os.makedirs(safe_path(out_dir), exist_ok=True)
    except PermissionError:
        warn(f"Could not create output folder (permission denied): {out_dir}")
        return

    summary_rows = []

    for code, (sensor_name, colour) in SENSOR_MAP.items():
        section(sensor_name)
        strain_col = f"{sensor_name}_Strain(\u00b5S)"
        if strain_col not in master.columns:
            warn(f"Column '{strain_col}' not found in master file - skipping.")
            continue

        d = master[["Timestamp", INDOOR_TEMP_COL, strain_col]].dropna()
        keep = exclude_ranges(d["Timestamp"], EXCLUDE_RANGES)
        if (~keep).sum():
            ok(f"Excluded {(~keep).sum()} rows in cabling-change window")
        d = d.loc[keep].sort_values("Timestamp").reset_index(drop=True)

        merged = pd.merge_asof(
            d, outside, on="Timestamp", direction="nearest",
            tolerance=pd.Timedelta(minutes=SNAP_TOLERANCE_MIN),
        ).dropna(subset=["Outside_Temp(\u00b0C)"])

        if len(merged) < 30:
            warn(f"{sensor_name}: too few aligned rows ({len(merged)}) - skipping.")
            continue
        ok(f"Aligned {len(merged)} rows within {SNAP_TOLERANCE_MIN}-minute tolerance")

        y = merged[strain_col].to_numpy()
        indoor = merged[INDOOR_TEMP_COL].to_numpy()
        outdoor = merged["Outside_Temp(\u00b0C)"].to_numpy()

        # ── Model A: indoor only ─────────────────────────────────────────
        X_a = sm.add_constant(indoor)
        model_a = sm.OLS(y, X_a).fit()
        r2_a = model_a.rsquared

        # ── Model B: indoor + outdoor ────────────────────────────────────
        X_b = sm.add_constant(np.column_stack([indoor, outdoor]))
        model_b = sm.OLS(y, X_b).fit()
        r2_b = model_b.rsquared
        r2_gain = r2_b - r2_a

        # F-test: does adding outdoor significantly improve the fit?
        f_stat, f_pvalue, _ = model_b.compare_f_test(model_a)

        # Standardized coefficients (z-score both predictors) for fair
        # comparison of relative effect size
        indoor_z = (indoor - indoor.mean()) / indoor.std()
        outdoor_z = (outdoor - outdoor.mean()) / outdoor.std()
        X_std = sm.add_constant(np.column_stack([indoor_z, outdoor_z]))
        model_std = sm.OLS(y, X_std).fit()
        std_coef_indoor, std_coef_outdoor = model_std.params[1], model_std.params[2]

        # VIF for the two predictors in Model B
        X_vif = np.column_stack([indoor, outdoor])
        vif_indoor = variance_inflation_factor(np.column_stack([np.ones(len(indoor)), X_vif]), 1)
        vif_outdoor = variance_inflation_factor(np.column_stack([np.ones(len(indoor)), X_vif]), 2)

        ok(f"R2 indoor-only: {r2_a:.4f}   R2 both: {r2_b:.4f}   Gain: {r2_gain:.4f}")
        ok(f"F-test for outdoor's added contribution: F={f_stat:.2f}  p={f_pvalue:.4g}")
        ok(f"Standardized coef - indoor: {std_coef_indoor:.4f}  outdoor: {std_coef_outdoor:.4f}")
        ok(f"VIF - indoor: {vif_indoor:.2f}  outdoor: {vif_outdoor:.2f}")

        summary_rows.append({
            "Sensor": sensor_name,
            "R2_indoor_only": r2_a, "R2_both": r2_b, "R2_Gain_from_Outdoor": r2_gain,
            "F_stat": f_stat, "F_pvalue": f_pvalue,
            "Raw_Coef_Indoor": model_b.params[1], "Raw_Coef_Outdoor": model_b.params[2],
            "Std_Coef_Indoor": std_coef_indoor, "Std_Coef_Outdoor": std_coef_outdoor,
            "Outdoor_p_value": model_b.pvalues[2],
            "VIF_Indoor": vif_indoor, "VIF_Outdoor": vif_outdoor,
            "n_rows": len(merged),
        })

        # ── Added-variable (partial regression) plot ─────────────────────
        strain_resid = y - model_a.predict(X_a)
        outdoor_on_indoor = sm.OLS(outdoor, sm.add_constant(indoor)).fit()
        outdoor_resid = outdoor - outdoor_on_indoor.predict(sm.add_constant(indoor))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=outdoor_resid, y=strain_resid, mode="markers",
            marker=dict(color=colour, size=3, opacity=0.35),
            name="Data (indoor effect removed)",
        ))
        line_x = np.array([outdoor_resid.min(), outdoor_resid.max()])
        line_y = model_b.params[2] * line_x
        fig.add_trace(go.Scatter(
            x=line_x, y=line_y, mode="lines",
            line=dict(color="black", width=2),
            name=f"Outdoor's unique effect (slope={model_b.params[2]:.3f})",
        ))
        fig.update_layout(
            title=f"{sensor_name}: Added-variable plot (outdoor, controlling for indoor)",
            font=dict(family=FONT, size=12),
            xaxis_title="Outdoor temperature (residual after removing indoor's effect)",
            yaxis_title="Strain (residual after removing indoor's effect)",
            margin=dict(l=65, r=40, t=50, b=60), plot_bgcolor="white",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=True, zerolinecolor="#cccccc")
        fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=True, zerolinecolor="#cccccc")

        html_path = safe_path(os.path.join(out_dir, f"{sensor_name}_added_variable_plot.html"))
        try:
            fig.write_html(html_path, include_plotlyjs="cdn")
            ok(f"Saved: {os.path.basename(html_path)}")
        except PermissionError:
            warn(f"Could not write plot (file open elsewhere?): {html_path}")

    if summary_rows:
        csv_path = safe_path(os.path.join(out_dir, "multiple_regression_summary.csv"))
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
