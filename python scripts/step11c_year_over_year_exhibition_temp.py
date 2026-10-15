"""
step11c_year_over_year_exhibition_temp.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 11c (standalone, NOT part of run_full_pipeline.py)
Overlays 2025 and 2026's indoor exhibition temperature on the same
calendar window (mid-June to mid-August), to see whether the ambient
conditions themselves were similar or different across the two years -
useful context for interpreting Step 11a/11b's strain comparisons.

WHERE THIS SITS
=================
Reads Step 1's master file. Standalone - not wired into run_full_pipeline.py.

OVERLAP WINDOW
================
14 June to 13 August (inclusive), by month-day, ignoring year - matches
Step 11a/11b.

WHAT THIS SCRIPT DOES
======================
One plot: 2025's indoor temperature (light colour) and 2026's (full colour)
overlaid on the same elapsed-day x-axis.

OUTPUTS (saved to Output\\Round 2\\Step 11 Year over Year Comparison\\):
  exhibition_temp_yoy.html

USAGE
=====
    Run AFTER step1_collation_alignment.py has produced its master file.
        python step11c_year_over_year_exhibition_temp.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds - save
edits as step11c_year_over_year_exhibition_temp_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import warnings

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
ROUND_NAME   = "Round 2"
STEP1_FOLDER = "Step 1 Collation Alignment and Raw Plots"
STEP_FOLDER  = "Step 11 Year over Year Comparison"

FONT = "Arial"
INDOOR_TEMP_COL = "Exhibition_Temp(\u00b0C)"

LIGHT_COLOUR = "#FFCC80"   # light orange, 2025
FULL_COLOUR  = "#E65100"   # full orange, 2026

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


def window_bounds(year):
    start = pd.Timestamp(year, *WINDOW_START_MONTHDAY)
    end = pd.Timestamp(year, *WINDOW_END_MONTHDAY, 23, 59, 59)
    return start, end


def elapsed_day_ticks():
    ref_start = pd.Timestamp(2001, *WINDOW_START_MONTHDAY)
    ref_end = pd.Timestamp(2001, *WINDOW_END_MONTHDAY)
    total_days = (ref_end - ref_start).days
    tickvals = list(range(0, total_days + 1, 7))
    if tickvals[-1] != total_days:
        tickvals.append(total_days)
    ticktext = [(ref_start + pd.Timedelta(days=d)).strftime("%d %b") for d in tickvals]
    return tickvals, ticktext


def main():
    section("Step 11c: Year-over-Year Exhibition Temperature")

    step1_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP1_FOLDER)
    master_path = safe_path(os.path.join(step1_dir, "All_Sensors_Aligned_Filled_Trimmed.csv"))
    if not os.path.exists(master_path):
        warn(f"Master file not found: {master_path}")
        return
    master = pd.read_csv(master_path, parse_dates=["Timestamp"])
    if INDOOR_TEMP_COL not in master.columns:
        warn(f"Temperature column '{INDOOR_TEMP_COL}' not found. Aborting.")
        return
    d = master[["Timestamp", INDOOR_TEMP_COL]].dropna()
    ok(f"Loaded indoor temperature: {len(d)} rows")

    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    try:
        os.makedirs(safe_path(out_dir), exist_ok=True)
    except PermissionError:
        warn(f"Could not create output folder (permission denied): {out_dir}")
        return

    year_styles = {YEARS[0]: (LIGHT_COLOUR, "2025"), YEARS[1]: (FULL_COLOUR, "2026")}

    fig = go.Figure()
    any_data = False
    for year in YEARS:
        start, end = window_bounds(year)
        yd = d[(d["Timestamp"] >= start) & (d["Timestamp"] <= end)].copy()
        if yd.empty:
            warn(f"{year}: no data in window - skipped.")
            continue
        any_data = True
        yd["Elapsed_Days"] = (yd["Timestamp"] - start).dt.total_seconds() / 86400
        colour, label = year_styles[year]
        fig.add_trace(go.Scatter(x=yd["Elapsed_Days"], y=yd[INDOOR_TEMP_COL], mode="lines",
                                 line=dict(color=colour, width=1.3), name=label,
                                 customdata=yd["Timestamp"],
                                 hovertemplate=f"{label}<br>%{{customdata|%d %b %Y %H:%M}}"
                                               f"<br>%{{y:.2f}} \u00b0C<extra></extra>"))
        ok(f"{year}: {len(yd)} rows plotted")

    if not any_data:
        warn("No data found in the overlap window for either year. Nothing saved.")
        return

    fig.update_layout(
        title=f"Exhibition Temperature: Year-over-Year "
              f"({pd.Timestamp(2000,*WINDOW_START_MONTHDAY).strftime('%d %b')}"
              f"\u2013{pd.Timestamp(2000,*WINDOW_END_MONTHDAY).strftime('%d %b')})",
        font=dict(family=FONT, size=12),
        xaxis_title="Date (both years overlaid on the same calendar position)",
        yaxis_title="Temperature (\u00b0C)",
        legend=dict(orientation="h", y=1.08, x=0, xanchor="left"),
        margin=dict(l=65, r=40, t=90, b=70), plot_bgcolor="white",
    )
    tickvals, ticktext = elapsed_day_ticks()
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext,
                     showgrid=True, gridcolor="#eeeeee", rangeslider_visible=True)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")

    html_path = safe_path(os.path.join(out_dir, "exhibition_temp_yoy.html"))
    try:
        fig.write_html(html_path, include_plotlyjs="cdn")
        ok(f"Saved: {os.path.basename(html_path)}")
    except PermissionError:
        warn(f"Could not write plot (file open elsewhere?): {html_path}")

    section("Done")


if __name__ == "__main__":
    main()
