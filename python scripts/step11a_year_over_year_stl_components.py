"""
step11a_year_over_year_stl_components.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 11a (standalone, NOT part of run_full_pipeline.py)
Overlays 2025 and 2026 on the same calendar window (mid-June to mid-August,
the period both years have in common) to see whether the trend, daily
seasonal cycle, and residual noise behaved similarly or differently across
the two years.

WHERE THIS SITS
=================
Reads Step 4's per-sensor decomposed CSVs. Standalone for now (per project
decision) - not wired into run_full_pipeline.py. May be added to the
pipeline later if it proves useful.

OVERLAP WINDOW
================
14 June to 13 August (inclusive), by month-day, ignoring year - the widest
window both 2025 and 2026 have real data for. Edit WINDOW_START/WINDOW_END
below if the data range changes in a later round.

HOW THE TWO YEARS ARE OVERLAID
=================================
Both years are re-expressed as "days elapsed since 14 June of that year",
so 2025's and 2026's curves share the same x-axis regardless of the actual
calendar date. 2025 is drawn in a lightened version of the sensor's colour,
2026 in the full colour, per the user's request.

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors, produces ONE file with three stacked
panels (Trend / Daily Seasonal / Residual), each showing 2025 and 2026
overlaid on the same elapsed-day x-axis.

OUTPUTS (saved to Output\\Round 2\\Step 11 Year over Year Comparison\\):
  <Sensor>_yoy_stl_components.html   - one file per sensor

USAGE
=====
    Run AFTER step4_stl_decomposition.py.
        python step11a_year_over_year_stl_components.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds - save
edits as step11a_year_over_year_stl_components_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import warnings

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
STEP_FOLDER  = "Step 11 Year over Year Comparison"

FONT = "Arial"

SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

WINDOW_START_MONTHDAY = (6, 14)    # 14 June
WINDOW_END_MONTHDAY   = (8, 13)    # 13 August
YEARS = [2025, 2026]

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


def lighten_hex(hex_colour, factor=0.65):
    """Blend a hex colour toward white by `factor` (0=no change, 1=white)."""
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def window_bounds(year):
    start = pd.Timestamp(year, *WINDOW_START_MONTHDAY)
    end = pd.Timestamp(year, *WINDOW_END_MONTHDAY, 23, 59, 59)
    return start, end


def elapsed_day_ticks():
    """Weekly tick marks across the window, labelled with real calendar-style
    dates (e.g. '14 Jun', '21 Jun') using a reference year so they read
    naturally, even though the same tick maps to both 2025 and 2026."""
    ref_start = pd.Timestamp(2001, *WINDOW_START_MONTHDAY)
    ref_end = pd.Timestamp(2001, *WINDOW_END_MONTHDAY)
    total_days = (ref_end - ref_start).days
    tickvals = list(range(0, total_days + 1, 7))
    if tickvals[-1] != total_days:
        tickvals.append(total_days)
    ticktext = [(ref_start + pd.Timedelta(days=d)).strftime("%d %b") for d in tickvals]
    return tickvals, ticktext


def extract_year_window(df, ts_col, year):
    start, end = window_bounds(year)
    d = df[(df[ts_col] >= start) & (df[ts_col] <= end)].copy()
    d["Elapsed_Days"] = (d[ts_col] - start).dt.total_seconds() / 86400
    return d


def main():
    section("Step 11a: Year-over-Year STL Component Comparison")

    step4_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP4_FOLDER)
    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    try:
        os.makedirs(safe_path(out_dir), exist_ok=True)
    except PermissionError:
        warn(f"Could not create output folder (permission denied): {out_dir}")
        return

    for code, (sensor_name, colour) in SENSOR_MAP.items():
        section(sensor_name)
        decomp_path = safe_path(os.path.join(step4_dir, f"{sensor_name}_decomposed.csv"))
        if not os.path.exists(decomp_path):
            warn(f"Decomposed file not found: {decomp_path} - skipping.")
            continue

        decomp = pd.read_csv(decomp_path, parse_dates=["Timestamp"], encoding="utf-8-sig")
        trend_col = f"{sensor_name}_Trend(\u00b5S)"
        seasonal_col = f"{sensor_name}_Seasonal_Daily(\u00b5S)"
        residual_col = f"{sensor_name}_Residual(\u00b5S)"
        required = {"Timestamp", trend_col, seasonal_col, residual_col}
        if not required.issubset(decomp.columns):
            warn(f"{sensor_name}: expected columns {required} not all found "
                 f"(has {list(decomp.columns)}) - skipping.")
            continue

        light_colour = lighten_hex(colour)
        year_styles = {YEARS[0]: (light_colour, "2025"), YEARS[1]: (colour, "2026")}

        panels = [("Trend", trend_col), ("Daily Seasonal", seasonal_col), ("Residual", residual_col)]
        fig = make_subplots(rows=3, cols=1, subplot_titles=[p[0] for p in panels],
                            vertical_spacing=0.08, shared_xaxes=True)

        any_data = False
        for row_i, (panel_name, col) in enumerate(panels, start=1):
            for year in YEARS:
                d = extract_year_window(decomp, "Timestamp", year).dropna(subset=[col])
                if d.empty:
                    warn(f"{panel_name} / {year}: no data in window - skipped.")
                    continue
                any_data = True
                line_colour, label = year_styles[year]
                fig.add_trace(
                    go.Scatter(x=d["Elapsed_Days"], y=d[col], mode="lines",
                              line=dict(color=line_colour, width=1.3), name=label,
                              legendgroup=label, showlegend=(row_i == 1),
                              customdata=d["Timestamp"],
                              hovertemplate=f"{label}<br>%{{customdata|%d %b %Y %H:%M}}"
                                            f"<br>%{{y:.3f}} \u00b5S<extra></extra>"),
                    row=row_i, col=1,
                )
            fig.update_yaxes(title_text=f"{panel_name} (\u00b5S)", row=row_i, col=1)

        if not any_data:
            warn(f"{sensor_name}: no data found in the overlap window for any panel - skipping file.")
            continue

        tickvals, ticktext = elapsed_day_ticks()
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext,
                         title_text="Date (both years overlaid on the same calendar position)",
                         rangeslider_visible=True, row=3, col=1)
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, row=1, col=1)
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, row=2, col=1)
        fig.update_layout(
            title=f"{sensor_name}: Year-over-Year STL Components "
                  f"({pd.Timestamp(2000,*WINDOW_START_MONTHDAY).strftime('%d %b')}"
                  f"\u2013{pd.Timestamp(2000,*WINDOW_END_MONTHDAY).strftime('%d %b')})",
            font=dict(family=FONT, size=11), height=820,
            legend=dict(orientation="h", y=1.05, x=0, xanchor="left"),
            margin=dict(l=65, r=40, t=90, b=60), plot_bgcolor="white",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
        fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")

        html_path = safe_path(os.path.join(out_dir, f"{sensor_name}_yoy_stl_components.html"))
        try:
            fig.write_html(html_path, include_plotlyjs="cdn")
            ok(f"Saved: {os.path.basename(html_path)}")
        except PermissionError:
            warn(f"Could not write plot (file open elsewhere?): {html_path}")

    section("Done")


if __name__ == "__main__":
    main()
