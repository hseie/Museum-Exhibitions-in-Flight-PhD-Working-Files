"""
step4_stl_decomposition.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 4
STL/MSTL decomposition of each sensor's temperature-compensated strain,
daily period only (96 steps = 24 hours at 15-minute resolution).

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors:
  1. Loads Step 3's compensated data for that sensor.
  2. Fills isolated remaining NaNs by linear interpolation (STL/MSTL
     requires a complete series) — the ORIGINAL, non-interpolated values
     are still what gets plotted for the "Original" panel, with the
     halftone-dashed-gap convention showing where data was actually
     missing.
  3. Runs MSTL with periods=[96] (daily only). The annual period is
     deliberately NOT fitted — under 2 full annual cycles are available
     in the dataset, so any annual signal remains folded into the trend.
     This is a documented limitation, not an error.
  4. Saves the decomposed series (original, trend, daily seasonal,
     residual) to CSV, and plots a 4-panel chart per sensor.

OUTPUTS (saved to Output\\Round 1\\Step 4 STL Decomposition\\):
  <Sensor>_decomposed.csv        - Timestamp, original, trend, seasonal_daily,
                                    seasonal_annual (NaN - not fitted), residual (x6)
  All_Sensors_Decomposed.csv     - all 6 sensors, merged
  decomposition_<Sensor>.html    - 4-panel plot: original / trend / daily
                                    seasonal / residual (x6)

USAGE
=====
    Run AFTER step3_temperature_compensation.py.
        python step4_stl_decomposition.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds — save
edits as step4_stl_decomposition_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL
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
ROUND_NAME   = "Round 1"
STEP3_FOLDER = "Step 3 Compensation"
STEP_FOLDER  = "Step 4 STL Decomposition"

FONT = "Arial"
STEPS_PER_DAY = 96   # 15-minute intervals in 24 hours

SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

SEASONAL_COLOUR  = "#607D8B"
RESIDUAL_COLOUR  = "#9E9E9E"

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


def get_step3_dir():
    return os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP3_FOLDER)


def get_output_dir():
    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    os.makedirs(safe_path(out_dir), exist_ok=True)
    return out_dir


def save_html(fig, path):
    fig.write_html(safe_path(path), include_plotlyjs="cdn")
    ok(f"Saved: {os.path.basename(path)}")


def month_ticks(t_min, t_max):
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    step3_dir = get_step3_dir()
    out_dir   = get_output_dir()
    print(f"Reading Step 3 output from: {step3_dir}")
    print(f"Output folder: {out_dir}")

    section("STEP 4 - STL/MSTL decomposition (daily period, per sensor)")

    all_decomposed = {}

    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        comp_path = os.path.join(step3_dir, f"{sensor_key}_compensated.csv")
        if not os.path.isfile(safe_path(comp_path)):
            warn(f"{sensor_key}: compensated file not found — skipped ({comp_path})")
            continue

        df = pd.read_csv(safe_path(comp_path), parse_dates=["Timestamp"])
        comp_col = f"{sensor_key}_Compensated(\u00b5S)"
        if comp_col not in df.columns:
            warn(f"{sensor_key}: '{comp_col}' not found — skipped")
            continue

        df = df.sort_values("Timestamp").reset_index(drop=True)
        series = df[comp_col]
        n_valid = series.notna().sum()

        if n_valid < STEPS_PER_DAY * 2:
            warn(f"{sensor_key}: fewer than 2 full daily cycles ({n_valid} valid steps) — skipped")
            continue

        # Restrict interpolation/decomposition to this sensor's own
        # genuinely active window (its first to last real reading).
        # The master file is an outer join across all sensors' date
        # ranges, so a sensor installed later (e.g. Starboard Top) has
        # real NaN rows stretching back before it existed — interpolating
        # with limit_direction='both' across the FULL series would
        # backfill that entire empty stretch with its first reading,
        # producing a fake flat trend/seasonal/residual before its
        # install date. Interpolating only within [first_valid, last_valid]
        # fixes isolated internal gaps (as intended) without inventing
        # data for a period before the sensor was there at all.
        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()
        active = series.loc[first_valid:last_valid]
        active_filled = active.interpolate(method="linear", limit_direction="both")

        try:
            result = MSTL(active_filled, periods=[STEPS_PER_DAY]).fit()
            trend_active = result.trend
            if hasattr(result.seasonal, "columns"):
                seas_cols = [c for c in result.seasonal.columns if str(STEPS_PER_DAY) in str(c)]
                seasonal_active = result.seasonal[seas_cols[0]] if seas_cols else result.seasonal.iloc[:, 0]
            else:
                seasonal_active = result.seasonal
            residual_active = result.resid
        except Exception as e:
            warn(f"{sensor_key}: MSTL failed ({e}) — skipped")
            continue

        # Reindex back to the sensor's full row set — NaN outside the
        # active window (before install / after last reading), matching
        # how the Original column already behaves.
        trend    = pd.Series(np.nan, index=series.index)
        seasonal = pd.Series(np.nan, index=series.index)
        residual = pd.Series(np.nan, index=series.index)
        trend.loc[first_valid:last_valid]    = trend_active.values
        seasonal.loc[first_valid:last_valid] = seasonal_active.values
        residual.loc[first_valid:last_valid] = residual_active.values

        out_df = pd.DataFrame({
            "Timestamp": df["Timestamp"],
            f"{sensor_key}_Original(\u00b5S)": series.values,       # real gaps intact
            f"{sensor_key}_Trend(\u00b5S)": trend.values,
            f"{sensor_key}_Seasonal_Daily(\u00b5S)": seasonal.values,
            f"{sensor_key}_Seasonal_Annual(\u00b5S)": np.nan,        # not fitted this round
            f"{sensor_key}_Residual(\u00b5S)": residual.values,
        })
        out_df.to_csv(
            safe_path(os.path.join(out_dir, f"{sensor_key}_decomposed.csv")),
            index=False, encoding="utf-8-sig",
        )
        ok(f"{sensor_key}: decomposed ({n_valid:,} valid steps) -> {sensor_key}_decomposed.csv")

        # Keep everything needed for plotting — plots are generated in a
        # second pass below, once the GLOBAL date span across all sensors
        # is known (see note there for why).
        all_decomposed[sensor_key] = dict(
            colour=colour, df=out_df, timestamp=df["Timestamp"],
            series=series, trend=trend, seasonal=seasonal, residual=residual,
        )

    if not all_decomposed:
        raise SystemExit("No sensor was decomposed — nothing further to do.")

    # ── Global date span across ALL sensors ─────────────────────────────
    # Used to force every sensor's plot onto the SAME x-axis span. Without
    # this, Starboard Top's plot (installed later, trimmed to start Sept
    # 2025) would auto-scale to its own shorter date range and look like a
    # different time window from the other five sensors when compared
    # side by side. Forcing the shared range means Starboard Top's plot
    # simply shows blank canvas before its install/trim date, rather than
    # a foreshortened axis.
    global_t_min = min(d["timestamp"].min() for d in all_decomposed.values())
    global_t_max = max(d["timestamp"].max() for d in all_decomposed.values())
    ticks = month_ticks(global_t_min, global_t_max)
    ticktext = [t.strftime("%d %b %Y") for t in ticks]

    section("Plots")

    for sensor_key, d in all_decomposed.items():
        colour = d["colour"]
        label = sensor_key.replace("_", " ")
        timestamp = d["timestamp"]

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=["Compensated (original)", "Trend", "Daily seasonal", "Residual"],
            vertical_spacing=0.06,
        )
        panels = [
            (d["series"], colour, "Original", False),   # real gaps left blank
            (d["trend"], colour, "Trend", True),
            (d["seasonal"], SEASONAL_COLOUR, "Daily seasonal", True),
            (d["residual"], RESIDUAL_COLOUR, "Residual", True),
        ]
        for row, (data, c, name, connect) in enumerate(panels, 1):
            fig.add_trace(go.Scatter(
                x=timestamp, y=data, mode="lines", name=name,
                line=dict(color=c, width=0.8), connectgaps=connect,
                showlegend=False,
            ), row=row, col=1)

        for row in range(1, 5):
            fig.update_xaxes(
                tickmode="array", tickvals=ticks,
                ticktext=ticktext if row == 4 else [""] * len(ticks),
                tickangle=-45, showgrid=True, gridcolor="#eeeeee",
                range=[global_t_min, global_t_max],
                row=row, col=1,
            )
        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.035), row=4, col=1,
        )
        fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", title_text="\u00b5S")

        fig.update_layout(
            title=dict(
                text=(f"{label} - STL Decomposition (daily period = 96 steps)<br>"
                      "<sup>Annual period not separable: fewer than 2 full cycles available. "
                      "Trend includes any annual seasonal response. "
                      "X-axis spans the full monitoring period across all sensors.</sup>"),
                font=dict(size=12, family=FONT), x=0.0, xanchor="left",
            ),
            height=780,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=65, r=40, t=95, b=110),
            font=dict(family=FONT),
        )
        save_html(fig, os.path.join(out_dir, f"decomposition_{sensor_key}.html"))

    # ── Combined output (all sensors merged) ────────────────────────────
    combined = None
    for sensor_key, d in all_decomposed.items():
        combined = d["df"] if combined is None else pd.merge(
            combined, d["df"], on="Timestamp", how="outer"
        )
    combined = combined.sort_values("Timestamp").reset_index(drop=True)
    combined.to_csv(
        safe_path(os.path.join(out_dir, "All_Sensors_Decomposed.csv")),
        index=False, encoding="utf-8-sig",
    )
    ok(f"All_Sensors_Decomposed.csv saved: {len(combined):,} rows")

    section("STEP 4 COMPLETE")
    print(f"  All outputs saved to:\n  {out_dir}")


if __name__ == "__main__":
    main()
