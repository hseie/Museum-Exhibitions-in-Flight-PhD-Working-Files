"""
step3_temperature_compensation.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 3
Temperature compensation of each sensor's strain, using the OLS
coefficients (alpha) fitted in Step 2.

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors:
  1. Loads Step 1's combined master file (strain + exhibition temperature,
     15-minute aligned, gap-filled, settling-trimmed) and Step 2's
     regression_summary.csv (alpha per sensor).
  2. Recalculates T0 (exhibition temperature period mean) FRESH on the
     full combined dataset — per the project's fixed convention, this is
     never carried over from a prior round.
  3. Applies the compensation formula:
         epsilon_normalised = epsilon_measured - alpha * (T - T0)
  4. Plots, per sensor, an overlay of raw strain (grey) vs compensated
     strain (sensor colour), both with the halftone-dashed-gap convention.
  5. Plots all six sensors' compensated strain together on one combined
     time-series plot.

OUTPUTS (saved to Output\\Round 1\\Step 3 Compensation\\):
  <Sensor>_compensated.csv           - Timestamp, raw strain, compensated strain (x6)
  All_Sensors_Compensated.csv        - all 6 sensors (raw + compensated) + temp, merged
  compensation_parameters.csv        - alpha, T0, beta per sensor (traceability)
  overlay_<Sensor>.html              - raw (grey) vs compensated overlay, per sensor (x6)
  compensated_timeseries_all_sensors.html - all sensors, compensated, one plot

USAGE
=====
    Run AFTER step1_collation_alignment.py and step2_temperature_regression.py.
        python step3_temperature_compensation.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds — save
edits as step3_temperature_compensation_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import warnings

import numpy as np
import pandas as pd
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
STEP2_FOLDER = "Step 2 Temperature Regression"
STEP_FOLDER  = "Step 3 Compensation"

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
RAW_GREY = "#aaaaaa"

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
    """Windows MAX_PATH (260-char) workaround — see Step 1 for full explanation."""
    if os.name == "nt":
        abs_path = os.path.abspath(path)
        if not abs_path.startswith("\\\\?\\"):
            abs_path = "\\\\?\\" + abs_path
        return abs_path
    return path


def get_step1_dir():
    return os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP1_FOLDER)


def get_step2_dir():
    return os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP2_FOLDER)


def get_output_dir():
    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    os.makedirs(safe_path(out_dir), exist_ok=True)
    return out_dir


def save_html(fig, path):
    fig.write_html(safe_path(path), include_plotlyjs="cdn")
    ok(f"Saved: {os.path.basename(path)}")


def month_ticks(t_min, t_max):
    """List of the 1st and 15th of every month spanned by [t_min, t_max]."""
    ticks = []
    cur = pd.Timestamp(t_min.year, t_min.month, 1)
    end = pd.Timestamp(t_max.year, t_max.month, 1)
    while cur <= end:
        ticks.append(cur)
        try:
            ticks.append(cur.replace(day=15))
        except ValueError:
            pass
        cur = (cur + pd.Timedelta(days=32)).replace(day=1)
    ticks = sorted(t for t in ticks if t_min - pd.Timedelta(days=1) <= t <= t_max + pd.Timedelta(days=1))
    return ticks


def apply_month_ticks(fig, t_min, t_max):
    ticks = month_ticks(t_min, t_max)
    fig.update_xaxes(
        tickmode="array", tickvals=ticks,
        ticktext=[t.strftime("%d %b %Y") for t in ticks], tickangle=-45,
    )


def plotly_base(title, subtitle="", height=560):
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>" if subtitle else title,
            font=dict(size=13, family=FONT), x=0.0, xanchor="left",
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#eeeeee", type="date",
            rangeslider=dict(visible=True, thickness=0.04),
        ),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee",
                    zeroline=True, zerolinecolor="#dddddd"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.38, x=0, xanchor="left",
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#cccccc", borderwidth=1),
        height=height,
        margin=dict(l=65, r=60, t=90, b=160),
        hovermode="x unified",
        font=dict(family=FONT),
    )
    return fig


def add_gapped_line(fig, x, y, name, colour, width=0.8, yaxis="y",
                     legendgroup=None, hovertemplate=None, dash_style="dash",
                     line_dash=None, opacity_main=1.0):
    """Solid real-data line + faded dashed halftone bridging the same gaps."""
    group = legendgroup or name
    main_line = dict(color=colour, width=width)
    if line_dash:
        main_line["dash"] = line_dash
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=name,
        line=main_line, opacity=opacity_main,
        connectgaps=False, legendgroup=group, yaxis=yaxis,
        hovertemplate=hovertemplate,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=f"{name} (gap)",
        line=dict(color=colour, width=width, dash=dash_style),
        connectgaps=True, opacity=0.3 * opacity_main, legendgroup=group, yaxis=yaxis,
        showlegend=False, hoverinfo="skip",
    ))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    step1_dir = get_step1_dir()
    step2_dir = get_step2_dir()
    out_dir   = get_output_dir()
    print(f"Reading Step 1 output from: {step1_dir}")
    print(f"Reading Step 2 output from: {step2_dir}")
    print(f"Output folder: {out_dir}")

    master_path = os.path.join(step1_dir, "All_Sensors_Aligned_Filled_Trimmed.csv")
    if not os.path.isfile(safe_path(master_path)):
        raise SystemExit(f"Could not find Step 1's master file:\n  {master_path}")
    master = pd.read_csv(safe_path(master_path), parse_dates=["Timestamp"])
    ok(f"Loaded master file: {len(master):,} rows")

    reg_path = os.path.join(step2_dir, "regression_summary.csv")
    if not os.path.isfile(safe_path(reg_path)):
        raise SystemExit(f"Could not find Step 2's regression_summary.csv:\n  {reg_path}")
    regression = pd.read_csv(safe_path(reg_path), encoding="utf-8-sig")
    alpha_col = [c for c in regression.columns if c.startswith("alpha")][0]
    alpha_by_label = dict(zip(regression["Sensor"], regression[alpha_col]))
    ok(f"Loaded regression coefficients for {len(alpha_by_label)} sensor(s)")

    if TEMP_COL not in master.columns:
        raise SystemExit(f"Expected temperature column '{TEMP_COL}' not found in master file.")

    # T0 recalculated fresh on the full combined dataset — fixed convention,
    # never reused from a prior round.
    T0 = float(master[TEMP_COL].dropna().mean())
    ok(f"T0 (period mean exhibition temperature) = {T0:.4f} \u00b0C, "
       f"n = {master[TEMP_COL].notna().sum():,}")

    section("STEP 3 - Temperature compensation (per sensor)")

    param_rows = []
    compensated_data = {}   # sensor_key -> dict(colour, df)

    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        comp_col   = f"{sensor_key}_Compensated(\u00b5S)"
        label = sensor_key.replace("_", " ")

        if strain_col not in master.columns:
            warn(f"{sensor_key}: strain column not found — skipped")
            continue
        if label not in alpha_by_label:
            warn(f"{sensor_key}: no regression coefficient found — skipped")
            continue

        alpha = float(alpha_by_label[label])
        df = master[["Timestamp", strain_col, TEMP_COL]].copy()
        df[comp_col] = df[strain_col] - alpha * (df[TEMP_COL] - T0)

        df[["Timestamp", strain_col, comp_col]].to_csv(
            safe_path(os.path.join(out_dir, f"{sensor_key}_compensated.csv")),
            index=False, encoding="utf-8-sig",
        )
        ok(f"{sensor_key}: compensated (alpha={alpha:.4f}, T0={T0:.4f}) -> "
           f"{sensor_key}_compensated.csv")

        param_rows.append({
            "Sensor": label,
            "alpha (\u00b5S/\u00b0C)": round(alpha, 4),
            "T0 (\u00b0C)": round(T0, 4),
            "n": int(df[[strain_col, TEMP_COL]].dropna().shape[0]),
        })

        compensated_data[sensor_key] = dict(colour=colour, df=df,
                                             strain_col=strain_col, comp_col=comp_col)

        # ── Per-sensor overlay plot: raw (grey) vs compensated (colour) ──
        fig = plotly_base(
            f"{label} - Raw vs Temperature-Compensated Strain",
            "Temperature effect removed via OLS coefficient from Step 2. Dashed = gap in the record.",
        )
        fig.update_yaxes(title_text="Strain (\u00b5S)")
        add_gapped_line(fig, df["Timestamp"], df[strain_col], "Raw strain", RAW_GREY, opacity_main=0.8)
        add_gapped_line(fig, df["Timestamp"], df[comp_col], "Compensated", colour, width=1.0)
        t_min_s, t_max_s = df["Timestamp"].min(), df["Timestamp"].max()
        apply_month_ticks(fig, t_min_s, t_max_s)
        save_html(fig, os.path.join(out_dir, f"overlay_{sensor_key}.html"))

    if not compensated_data:
        raise SystemExit("No sensor was compensated — nothing further to do.")

    # ── Combined master output (all sensors, raw + compensated + temp) ────
    combined = master[["Timestamp", TEMP_COL]].copy()
    for sensor_key, d in compensated_data.items():
        combined = combined.merge(
            d["df"][["Timestamp", d["strain_col"], d["comp_col"]]],
            on="Timestamp", how="left",
        )
    combined.to_csv(
        safe_path(os.path.join(out_dir, "All_Sensors_Compensated.csv")),
        index=False, encoding="utf-8-sig",
    )
    ok(f"All_Sensors_Compensated.csv saved: {len(combined):,} rows")

    pd.DataFrame(param_rows).to_csv(
        safe_path(os.path.join(out_dir, "compensation_parameters.csv")),
        index=False, encoding="utf-8-sig",
    )
    ok("compensation_parameters.csv saved")

    # ── Combined plot: all sensors, compensated, one chart ─────────────────
    section("Combined compensated plot")
    t_min = min(d["df"]["Timestamp"].min() for d in compensated_data.values())
    t_max = max(d["df"]["Timestamp"].max() for d in compensated_data.values())

    fig_all = plotly_base(
        "All Sensors - Compensated Strain (15-minute aligned)",
        "Temperature effect removed. Dashed = gap in the record.",
    )
    fig_all.update_yaxes(title_text="Compensated Strain (\u00b5S)")
    for sensor_key, d in compensated_data.items():
        label = sensor_key.replace("_", " ")
        add_gapped_line(
            fig_all, d["df"]["Timestamp"], d["df"][d["comp_col"]], label, d["colour"],
            hovertemplate=f"{label}<br>%{{x|%d %b %Y %H:%M}}<br>%{{y:.3f}} \u00b5S<extra></extra>",
        )
    apply_month_ticks(fig_all, t_min, t_max)
    save_html(fig_all, os.path.join(out_dir, "compensated_timeseries_all_sensors.html"))

    section("STEP 3 COMPLETE")
    print(f"  All outputs saved to:\n  {out_dir}")


if __name__ == "__main__":
    main()
