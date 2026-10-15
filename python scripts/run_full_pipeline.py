"""
run_full_pipeline.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — FULL PIPELINE (Steps 1-7, incremental)

WHAT THIS SCRIPT IS FOR
=========================
Designed to be handed off: someone adds new sensor export files (a new
numbered file, e.g. "NP 12.xlsx", OR more rows appended to the existing
latest file, e.g. "NP 11.xlsx" grows) into the same Sensors\\<X>\\ folders,
and updated exhibition temperature CSV into Exhibition Environment\\, then
just runs this one script. No other setup needed.

HOW ROUND DETECTION WORKS
============================
A persistent SEED file (seed_aligned_filled_trimmed.csv, living directly
in BASE_DIR — NOT inside an Output\\Round N\\ folder) holds the full
history of 15-minute aligned, gap-filled, settling-trimmed strain (all
sensors) and exhibition temperature, built up across every round so far.

On each run:
  1. Every raw sensor Excel file is read in full (all of them, every
     run — not just "new" ones by filename). This is deliberate: a file
     can be updated EITHER by adding a new numbered file OR by more rows
     being appended to the existing latest file in place, and there is
     no reliable way to tell those apart from the filename alone. Reading
     everything and filtering by timestamp afterward handles both cases
     correctly. (This does cost some read time on unchanged historical
     files each run — the calamine engine keeps that reasonable; see
     Step 1 for the measured numbers.)
  2. For each sensor (and for temperature), rows newer than the seed's
     last recorded timestamp are identified.
  3. If NOTHING is newer anywhere, the script prints a message and exits
     — no folder created, no files touched.
  4. If ANYTHING is newer for ANY sensor, a new Output\\Round N\\ folder
     is created (N = existing round-folder count + 1), a continuity
     check runs (see below), the seed is extended with the new data and
     OVERWRITTEN, and all seven steps run fresh on the now-larger
     dataset, writing into Round N.

Only Step 1's raw STRAIN collation benefits from this incremental
handling (it's the expensive part — large Excel sheets at ~15-second
resolution over many months). Burst collation and exhibition temperature
are cheap to fully re-read every round, so they are not treated
incrementally — they're just freshly recollated in full each time, same
as the standalone Step 1 script. Steps 2-7 always recompute from scratch
on the full current dataset each round (matching the project's fixed
convention: regression coefficients, T0, STL decomposition, etc. are
never carried over from a prior round).

CONTINUITY CHECK (gap / overlap between old and new data)
=============================================================
Per sensor and for temperature, this is a WARNING-ONLY check (never
blocks the run):
  - GAP: if the new data's earliest timestamp is more than 2 hours after
    the seed's last recorded timestamp, a gap is flagged — the raw
    export may be missing a stretch of history between rounds.
  - OVERLAP / DISCREPANCY: the tail of the newly-read raw data (the 2
    days before the seed's last timestamp) is aligned and compared
    against what the seed already has recorded for those same
    timestamps. If values differ by more than 1 microstrain, that's
    flagged — it suggests the source export revised historical values
    rather than only adding new ones.
All warnings are printed and written to round_summary.txt; none of them
stop the pipeline.

OUTPUTS
========
  BASE_DIR\\seed_aligned_filled_trimmed.csv   - persistent, overwritten each round
  BASE_DIR\\seed_meta.csv                      - per-sensor processing watermark
                                                  (true aligned-grid extent, not
                                                  "last valid value" — see
                                                  load_seed_meta() for why that
                                                  distinction matters)
  BASE_DIR\\Output\\Round N\\
      Step 1 Collation Alignment and Raw Plots\\
      Step 2 Temperature Regression\\
      Step 3 Compensation\\
      Step 4 STL Decomposition\\
      Step 5 Trend Analysis\\
      Step 6a Correlation Analysis (Compensated Strain)\\
      Step 6b Correlation Analysis (Trend)\\
      Step 7 Trend Model Fitting\\
      round_summary.txt   - rows added, warnings, per-sensor stats for this round

USAGE
=====
    python run_full_pipeline.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds — save
edits as run_full_pipeline_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.tsa.seasonal import MSTL
from statsmodels.nonparametric.smoothers_lowess import lowess
import ruptures
import openpyxl
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

try:
    import python_calamine  # noqa: F401
    EXCEL_ENGINE = "calamine"
except ImportError:
    EXCEL_ENGINE = "openpyxl"

# ══════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION — edit this section only
# ══════════════════════════════════════════════════════════════════════════

BASE_DIR = (
    r"C:\Users\c2051290\OneDrive - Newcastle University\NCLU\PhD\My PhD"
    r"\01 Onsite Work and Engineering\01 Sensors\02 Canberra analysis"
    r"\06 Files for Python Final Round"
)

SENSORS_SUBDIR    = "Sensors"
EXHIBITION_SUBDIR = "Exhibition Environment"
OUTPUT_BASE       = "Output"
SEED_FILE      = "seed_aligned_filled_trimmed.csv"
SEED_META_FILE = "seed_meta.csv"

EVENTS_FILE = os.path.join(
    BASE_DIR, "Exhibition Environment", "01 Canberra Recorded Events.xlsx"
)

FONT = "Arial"

SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

SETTLING_CUTOFF = {
    "NP": "2025-06-14", "NS": "2025-06-14", "PB": "2025-06-14",
    "PT": "2025-06-14", "SB": "2025-06-14", "ST": "2025-09-27",
}

EVENT_CATEGORY_COLOURS = {
    "weather": "#ED7D31", "around_aircraft": "#4472C4", "interference": "#FFC000",
}
THEME_INDEX_TO_CATEGORY = {5: "weather", 8: "around_aircraft", 7: "interference"}

# Multi-day event spans (shaded regions) are detected automatically from
# the sheet itself — see load_recorded_events()'s block_id and the
# grouping logic in Step 5's Plot 2: a category appearing more than once
# in a run of dated rows with no blank row between them is treated as
# one continuous span from its earliest to latest date in that run.
# Nothing to configure here; it follows whatever's in the sheet.

SNAP_TOLERANCE_MIN = 8
GAP_FILL_MAX_STEPS = 40
GAP_FILL_MAX_EDGE  = 5.0
STEPS_PER_DAY  = 96
ROLLING_DAYS   = 30
TEMP_COL = "Exhibition_Temp(\u00b0C)"

CONTINUITY_GAP_HOURS      = 2.0     # flag if new data starts more than this after seed's last point
CONTINUITY_OVERLAP_TOL_UE = 1.0     # microstrain tolerance for overlap-discrepancy check

# Confirmed instrumentation artifact (cabling change) - excluded ONLY from
# Steps 4b/4c below, which look at day-level noise/timing patterns where a
# known non-thermal disturbance would distort the result. Not applied
# anywhere else in the pipeline.
STEP4B4C_EXCLUDE_RANGES = [
    ("2026-07-21 00:00", "2026-07-23 17:00"),
]

PAIR_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]
NON_BEST_COLOURS = [
    "#A5D6A7", "#90CAF9", "#FFCC80", "#CE93D8",
    "#F48FB1", "#80DEEA", "#BCAAA4", "#EF9A9A",
]

# ══════════════════════════════════════════════════════════════════════════
# END USER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# Small shared helpers
# ══════════════════════════════════════════════════════════════════════════

def section(msg):
    print(f"\n{'='*70}\n  {msg}\n{'='*70}")


def subsection(msg):
    print(f"\n{'-'*70}\n  {msg}\n{'-'*70}")


def ok(msg):
    print(f"  OK  {msg}")


def warn(msg):
    print(f"  !!  {msg}")


def safe_path(path):
    """Windows MAX_PATH (260-char) workaround — see project notes."""
    if os.name == "nt":
        abs_path = os.path.abspath(path)
        if not abs_path.startswith("\\\\?\\"):
            abs_path = "\\\\?\\" + abs_path
        return abs_path
    return path


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


def apply_month_ticks(fig, t_min, t_max):
    ticks = month_ticks(t_min, t_max)
    fig.update_xaxes(tickmode="array", tickvals=ticks,
                      ticktext=[t.strftime("%d %b %Y") for t in ticks], tickangle=-45)


def plotly_base(title, subtitle="", height=560):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>" if subtitle else title,
                   font=dict(size=13, family=FONT), x=0.0, xanchor="left"),
        xaxis=dict(showgrid=True, gridcolor="#eeeeee", type="date",
                   rangeslider=dict(visible=True, thickness=0.04)),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee", zeroline=True, zerolinecolor="#dddddd"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.38, x=0, xanchor="left", font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.95)", bordercolor="#cccccc", borderwidth=1),
        height=height, margin=dict(l=65, r=60, t=90, b=160),
        hovermode="x unified", font=dict(family=FONT),
    )
    return fig


def add_gapped_line(fig, x, y, name, colour, width=0.8, yaxis="y", legendgroup=None,
                     hovertemplate=None, dash_style="dash", line_dash=None):
    group = legendgroup or name
    main_line = dict(color=colour, width=width)
    if line_dash:
        main_line["dash"] = line_dash
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=main_line,
                              connectgaps=False, legendgroup=group, yaxis=yaxis,
                              hovertemplate=hovertemplate))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{name} (gap)",
                              line=dict(color=colour, width=width, dash=dash_style),
                              connectgaps=True, opacity=0.3, legendgroup=group, yaxis=yaxis,
                              showlegend=False, hoverinfo="skip"))


def sig_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ══════════════════════════════════════════════════════════════════════════
# Step 1 logic: raw file reading, alignment, gap-fill, trim, temperature
# ══════════════════════════════════════════════════════════════════════════

def classify_sheet(xls, sheet_name):
    """Classify a sheet by its actual column header text — not name/position."""
    try:
        header_df = pd.read_excel(xls, sheet_name=sheet_name, nrows=0)
    except Exception:
        return None
    cols = " ".join(str(c) for c in header_df.columns).lower()
    if "high_rate" in cols or "high rate" in cols:
        return "strain"
    if "triggered" in cols or "trigger" in cols or "burst" in cols:
        return "burst"
    if "internal" in cols or "temperature" in cols:
        return "internal"
    return None


def load_sensor_files(sensor_folder_path, sensor_key):
    """Reads every Excel file in a sensor's folder, returns (strain_df, burst_df)."""
    files = sorted(
        glob.glob(os.path.join(sensor_folder_path, "*.xlsx")) +
        glob.glob(os.path.join(sensor_folder_path, "*.xls"))
    )
    if not files:
        warn(f"No Excel files found in {sensor_folder_path}")

    strain_col = f"{sensor_key}_Strain(\u00b5S)"
    burst_col  = f"{sensor_key}_Burst(\u00b5S)"
    strain_frames, burst_frames = [], []

    for f in files:
        try:
            xls = pd.ExcelFile(f, engine=EXCEL_ENGINE)
        except Exception as e:
            warn(f"Could not open {os.path.basename(f)}: {e}")
            continue

        sheet_high = sheet_burst = None
        for name in xls.sheet_names:
            kind = classify_sheet(xls, name)
            if kind == "strain" and sheet_high is None:
                sheet_high = name
            elif kind == "burst" and sheet_burst is None:
                sheet_burst = name

        if sheet_high is not None:
            df = pd.read_excel(xls, sheet_name=sheet_high)
            df = df.iloc[:, :2].copy()
            df.columns = ["Timestamp", strain_col]
            strain_frames.append(df)
        if sheet_burst is not None:
            df = pd.read_excel(xls, sheet_name=sheet_burst)
            df = df.iloc[:, :2].copy()
            df.columns = ["Timestamp", burst_col]
            burst_frames.append(df)

    def combine(frames, val_col):
        if not frames:
            return pd.DataFrame(columns=["Timestamp", val_col])
        out = pd.concat(frames, ignore_index=True)
        out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
        out = out.dropna(subset=["Timestamp"])
        out = out.drop_duplicates(subset=["Timestamp"])
        out = out.sort_values("Timestamp").reset_index(drop=True)
        return out

    strain_df = combine(strain_frames, strain_col)
    burst_df  = combine(burst_frames, burst_col)
    ok(f"{sensor_key}: {len(strain_df):,} raw strain rows, "
       f"{len(burst_df):,} raw burst rows read from {len(files)} file(s)")
    return strain_df, burst_df


def align_to_15min(df, ts_col, val_col, tolerance_min=SNAP_TOLERANCE_MIN,
                    grid_start=None, grid_end=None):
    d = df[[ts_col, val_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame(columns=["Timestamp", val_col])
    d = d.sort_values(ts_col)
    d[ts_col] = d[ts_col].dt.round("s")
    t_min = grid_start if grid_start is not None else d[ts_col].min().floor("15min")
    t_max = grid_end if grid_end is not None else d[ts_col].max().ceil("15min")
    if t_min > t_max:
        return pd.DataFrame(columns=["Timestamp", val_col])
    grid = pd.DataFrame({"Timestamp": pd.date_range(t_min, t_max, freq="15min")})
    merged = pd.merge_asof(
        grid, d.rename(columns={ts_col: "Timestamp", val_col: "value"}),
        on="Timestamp", tolerance=pd.Timedelta(minutes=tolerance_min), direction="nearest",
    )
    return merged.rename(columns={"value": val_col})


def gap_fill(series, max_steps=GAP_FILL_MAX_STEPS, max_edge=GAP_FILL_MAX_EDGE):
    s = series.copy()
    nan_mask = s.isna()
    if not nan_mask.any():
        return s
    changes = nan_mask.ne(nan_mask.shift()).cumsum()
    for _, grp in s.groupby(changes):
        if not grp.isna().all():
            continue
        idx = grp.index
        start, end = idx[0], idx[-1]
        n = len(idx)
        if n > max_steps:
            continue
        before_val = s.iloc[start - 1] if start > 0 else np.nan
        after_val  = s.iloc[end + 1] if end < len(s) - 1 else np.nan
        if pd.isna(before_val) or pd.isna(after_val):
            continue
        if abs(after_val - before_val) > max_edge:
            continue
        mid = n // 2
        s.iloc[start:start + mid]   = before_val
        s.iloc[start + mid:end + 1] = after_val
    return s


def apply_settling_trim(df, ts_col, cutoff_str):
    cutoff = pd.Timestamp(cutoff_str)
    return df[df[ts_col] >= cutoff].reset_index(drop=True)


def summarize_burst_events(df, ts_col, val_col, gap_seconds=2.0):
    if df.empty:
        return df
    d = df[[ts_col, val_col]].dropna().sort_values(ts_col).reset_index(drop=True)
    gaps = d[ts_col].diff().dt.total_seconds()
    event_id = (gaps > gap_seconds).cumsum()
    idx = d.groupby(event_id)[val_col].apply(lambda s: s.abs().idxmax())
    return d.loc[idx].reset_index(drop=True)


def load_exhibition_temp(exhibition_dir):
    candidates = [
        f for f in glob.glob(os.path.join(exhibition_dir, "*.csv"))
        if "temp" in os.path.basename(f).lower() and "rh" not in os.path.basename(f).lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No exhibition temperature CSV found in {exhibition_dir}")
    f = candidates[0]
    df = pd.read_csv(f, encoding="utf-8-sig")
    df.columns = ["Timestamp", TEMP_COL]
    ts_clean = (df["Timestamp"].astype(str).str.strip()
                .str.replace(r"\s+[A-Za-z]{2,4}$", "", regex=True))
    parsed = pd.to_datetime(ts_clean, format="%d-%b-%y %I:%M:%S %p", errors="coerce")
    if parsed.isna().mean() > 0.01:
        parsed = pd.to_datetime(ts_clean, errors="coerce", dayfirst=True)
    df["Timestamp"] = parsed
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    ok(f"Exhibition temperature loaded: {len(df):,} rows from {os.path.basename(f)}")
    return df


def load_recorded_events(path):
    if not os.path.isfile(safe_path(path)):
        warn(f"Recorded-events file not found, event highlighting skipped:\n      {path}")
        return []
    try:
        wb = openpyxl.load_workbook(safe_path(path), data_only=True)
    except PermissionError:
        warn(f"Recorded-events file is locked (probably open in Excel) — "
             f"event highlighting skipped. Close it in Excel and re-run "
             f"if you want events on the plot:\n      {path}")
        return []
    except Exception as e:
        warn(f"Could not read recorded-events file ({e}) — event highlighting skipped:\n      {path}")
        return []
    ws = wb.worksheets[0]
    # Blocks of consecutive dated rows with NO blank row between them
    # represent one continuous multi-day event (per the sheet's own
    # convention) — tag each event with a block_id so the plotting code
    # can group them. A row with text but no date (a wrapped-text
    # continuation line) doesn't break a block; only a fully blank row
    # does.
    events = []
    block_id = 0
    block_has_content = False
    for row in ws.iter_rows(min_row=4):
        row_is_blank = all(c.value is None for c in row)
        if row_is_blank:
            if block_has_content:
                block_id += 1
                block_has_content = False
            continue
        block_has_content = True
        date_cell = row[0]
        if date_cell.value is None or not hasattr(date_cell.value, "year"):
            continue
        date = pd.Timestamp(date_cell.value)
        fill = date_cell.fill
        theme = fill.fgColor.theme if (fill and fill.fgColor and fill.fgColor.type == "theme") else None
        category = THEME_INDEX_TO_CATEGORY.get(theme)
        if category is None:
            continue
        texts = [str(c.value).strip() for c in row if isinstance(c.value, str) and c.value.strip()]
        label = " / ".join(texts) if texts else "(no description)"
        events.append({"date": date, "label": label, "category": category, "block_id": block_id})
    ok(f"Recorded events loaded: {len(events)} dated row(s) categorised")
    return events


# ══════════════════════════════════════════════════════════════════════════
# Step 2 logic: temperature regression
# ══════════════════════════════════════════════════════════════════════════

def run_step2(master, out_dir):
    subsection("STEP 2 - Temperature regression (OLS, per sensor)")
    summary_rows = []
    alpha_by_sensor = {}

    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        label = sensor_key.replace("_", " ")
        if strain_col not in master.columns:
            warn(f"{sensor_key}: column not found — skipped")
            continue
        mask = master[strain_col].notna() & master[TEMP_COL].notna()
        n = int(mask.sum())
        if n < 30:
            warn(f"{sensor_key}: only {n} valid rows — skipped")
            continue

        T = master.loc[mask, TEMP_COL].to_numpy()
        S = master.loc[mask, strain_col].to_numpy()
        dates = master.loc[mask, "Timestamp"]
        slope, intercept, r_val, p_val, _ = stats.linregress(T, S)
        r2 = r_val ** 2
        S_hat = slope * T + intercept
        residual_se = float(np.sqrt(np.sum((S - S_hat) ** 2) / (n - 2)))
        days_since_start = (dates - dates.min()).dt.total_seconds() / 86400.0

        alpha_by_sensor[sensor_key] = slope
        summary_rows.append({
            "Sensor": label, "alpha (\u00b5S/\u00b0C)": round(slope, 4),
            "beta (\u00b5S)": round(intercept, 4), "R2": round(r2, 4),
            "SE": round(residual_se, 4), "p-value": p_val, "N": n,
        })
        ok(f"{sensor_key}: alpha={slope:.4f}  beta={intercept:.4f}  R2={r2:.4f}  n={n:,}")

        T_range = np.linspace(T.min(), T.max(), 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=T, y=S, mode="markers",
            marker=dict(color=days_since_start, colorscale="Viridis", size=3, opacity=0.45,
                        colorbar=dict(title=dict(text="Days since monitoring start", side="right"),
                                      thickness=14, len=0.75, y=0.55, yanchor="middle"),
                        showscale=True),
            name="Observed", hovertemplate="T=%{x:.2f} \u00b0C<br>\u03b5=%{y:.3f} \u00b5S<extra></extra>",
        ))
        fig.add_trace(go.Scatter(x=T_range, y=slope * T_range + intercept, mode="lines",
                                  name="OLS fit", line=dict(color=colour, width=2.5)))
        eq_text = f"\u03b5 = {slope:.4f} \u00b7 T + {intercept:.4f}<br>R\u00b2 = {r2:.4f}     n = {n:,}"
        fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.97, text=eq_text,
                            showarrow=False, font=dict(size=13, family=FONT, color="#222222"),
                            align="left", bgcolor="rgba(255,255,255,0.88)", bordercolor="#bbbbbb",
                            borderwidth=1, borderpad=6, xanchor="left", yanchor="top")
        fig.update_layout(
            title=dict(text=f"{label} - Strain vs Exhibition Temperature (OLS regression)",
                      font=dict(size=14, family=FONT), x=0.0, xanchor="left"),
            xaxis=dict(title="Exhibition Temperature (\u00b0C)", showgrid=True, gridcolor="#eeeeee"),
            yaxis=dict(title="Strain (\u00b5S)", showgrid=True, gridcolor="#eeeeee"),
            plot_bgcolor="white", paper_bgcolor="white", height=640,
            legend=dict(orientation="h", y=-0.14, x=0, xanchor="left", font=dict(size=11),
                        bgcolor="rgba(255,255,255,0.95)", bordercolor="#cccccc", borderwidth=1),
            margin=dict(l=70, r=110, t=70, b=100), font=dict(family=FONT),
        )
        save_html(fig, os.path.join(out_dir, f"regression_{sensor_key}.html"))

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            safe_path(os.path.join(out_dir, "regression_summary.csv")),
            index=False, encoding="utf-8-sig",
        )
        ok("regression_summary.csv saved")
    return alpha_by_sensor


# ══════════════════════════════════════════════════════════════════════════
# Step 3 logic: temperature compensation
# ══════════════════════════════════════════════════════════════════════════

def run_step3(master, alpha_by_sensor, out_dir):
    subsection("STEP 3 - Temperature compensation")
    T0 = float(master[TEMP_COL].dropna().mean())
    ok(f"T0 (period mean exhibition temperature) = {T0:.4f} \u00b0C")

    compensated_data = {}
    param_rows = []

    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        comp_col   = f"{sensor_key}_Compensated(\u00b5S)"
        label = sensor_key.replace("_", " ")
        if strain_col not in master.columns or sensor_key not in alpha_by_sensor:
            warn(f"{sensor_key}: missing strain column or regression coefficient — skipped")
            continue

        alpha = alpha_by_sensor[sensor_key]
        df = master[["Timestamp", strain_col, TEMP_COL]].copy()
        df[comp_col] = df[strain_col] - alpha * (df[TEMP_COL] - T0)
        df[["Timestamp", strain_col, comp_col]].to_csv(
            safe_path(os.path.join(out_dir, f"{sensor_key}_compensated.csv")),
            index=False, encoding="utf-8-sig",
        )
        ok(f"{sensor_key}: compensated (alpha={alpha:.4f}, T0={T0:.4f})")
        param_rows.append({"Sensor": label, "alpha (\u00b5S/\u00b0C)": round(alpha, 4),
                            "T0 (\u00b0C)": round(T0, 4),
                            "n": int(df[[strain_col, TEMP_COL]].dropna().shape[0])})
        compensated_data[sensor_key] = dict(colour=colour, df=df, strain_col=strain_col, comp_col=comp_col)

        fig = plotly_base(f"{label} - Raw vs Temperature-Compensated Strain",
                          "Temperature effect removed via OLS coefficient from Step 2. Dashed = gap in the record.")
        fig.update_yaxes(title_text="Strain (\u00b5S)")
        add_gapped_line(fig, df["Timestamp"], df[strain_col], "Raw strain", "#aaaaaa", width=0.8)
        add_gapped_line(fig, df["Timestamp"], df[comp_col], "Compensated", colour, width=1.0)
        t_min_s, t_max_s = df["Timestamp"].min(), df["Timestamp"].max()
        apply_month_ticks(fig, t_min_s, t_max_s)
        save_html(fig, os.path.join(out_dir, f"overlay_{sensor_key}.html"))

    if not compensated_data:
        return None, {}

    combined = master[["Timestamp", TEMP_COL]].copy()
    for sensor_key, d in compensated_data.items():
        combined = combined.merge(d["df"][["Timestamp", d["strain_col"], d["comp_col"]]],
                                   on="Timestamp", how="left")
    combined.to_csv(safe_path(os.path.join(out_dir, "All_Sensors_Compensated.csv")),
                     index=False, encoding="utf-8-sig")
    ok(f"All_Sensors_Compensated.csv saved: {len(combined):,} rows")
    pd.DataFrame(param_rows).to_csv(safe_path(os.path.join(out_dir, "compensation_parameters.csv")),
                                     index=False, encoding="utf-8-sig")

    t_min = min(d["df"]["Timestamp"].min() for d in compensated_data.values())
    t_max = max(d["df"]["Timestamp"].max() for d in compensated_data.values())
    fig_all = plotly_base("All Sensors - Compensated Strain (15-minute aligned)",
                          "Temperature effect removed. Dashed = gap in the record.")
    fig_all.update_yaxes(title_text="Compensated Strain (\u00b5S)")
    for sensor_key, d in compensated_data.items():
        label = sensor_key.replace("_", " ")
        add_gapped_line(fig_all, d["df"]["Timestamp"], d["df"][d["comp_col"]], label, d["colour"],
                        hovertemplate=f"{label}<br>%{{x|%d %b %Y %H:%M}}<br>%{{y:.3f}} \u00b5S<extra></extra>")
    apply_month_ticks(fig_all, t_min, t_max)
    save_html(fig_all, os.path.join(out_dir, "compensated_timeseries_all_sensors.html"))

    return combined, compensated_data


# ══════════════════════════════════════════════════════════════════════════
# Step 4 logic: STL/MSTL decomposition
# ══════════════════════════════════════════════════════════════════════════

def run_step4(compensated_data, out_dir):
    subsection("STEP 4 - STL/MSTL decomposition (daily period, per sensor)")
    all_decomposed = {}

    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        if sensor_key not in compensated_data:
            continue
        d = compensated_data[sensor_key]
        df = d["df"].sort_values("Timestamp").reset_index(drop=True)
        comp_col = d["comp_col"]
        series = df[comp_col]
        n_valid = series.notna().sum()
        if n_valid < STEPS_PER_DAY * 2:
            warn(f"{sensor_key}: fewer than 2 full daily cycles — skipped")
            continue

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

        trend    = pd.Series(np.nan, index=series.index)
        seasonal = pd.Series(np.nan, index=series.index)
        residual = pd.Series(np.nan, index=series.index)
        trend.loc[first_valid:last_valid]    = trend_active.values
        seasonal.loc[first_valid:last_valid] = seasonal_active.values
        residual.loc[first_valid:last_valid] = residual_active.values

        out_df = pd.DataFrame({
            "Timestamp": df["Timestamp"],
            f"{sensor_key}_Original(\u00b5S)": series.values,
            f"{sensor_key}_Trend(\u00b5S)": trend.values,
            f"{sensor_key}_Seasonal_Daily(\u00b5S)": seasonal.values,
            f"{sensor_key}_Seasonal_Annual(\u00b5S)": np.nan,
            f"{sensor_key}_Residual(\u00b5S)": residual.values,
        })
        out_df.to_csv(safe_path(os.path.join(out_dir, f"{sensor_key}_decomposed.csv")),
                       index=False, encoding="utf-8-sig")
        ok(f"{sensor_key}: decomposed ({n_valid:,} valid steps)")
        all_decomposed[sensor_key] = dict(colour=colour, df=out_df, timestamp=df["Timestamp"],
                                           series=series, trend=trend, seasonal=seasonal, residual=residual)

    if not all_decomposed:
        return {}, None

    global_t_min = min(dd["timestamp"].min() for dd in all_decomposed.values())
    global_t_max = max(dd["timestamp"].max() for dd in all_decomposed.values())
    ticks = month_ticks(global_t_min, global_t_max)
    ticktext = [t.strftime("%d %b %Y") for t in ticks]

    from plotly.subplots import make_subplots
    for sensor_key, dd in all_decomposed.items():
        colour = dd["colour"]
        label = sensor_key.replace("_", " ")
        timestamp = dd["timestamp"]
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=["Compensated (original)", "Trend", "Daily seasonal", "Residual"],
                            vertical_spacing=0.06)
        panels = [(dd["series"], colour, "Original", False), (dd["trend"], colour, "Trend", True),
                  (dd["seasonal"], "#607D8B", "Daily seasonal", True), (dd["residual"], "#9E9E9E", "Residual", True)]
        for row, (data, c, name, connect) in enumerate(panels, 1):
            fig.add_trace(go.Scatter(x=timestamp, y=data, mode="lines", name=name,
                                      line=dict(color=c, width=0.8), connectgaps=connect,
                                      showlegend=False), row=row, col=1)
        for row in range(1, 5):
            fig.update_xaxes(tickmode="array", tickvals=ticks,
                              ticktext=ticktext if row == 4 else [""] * len(ticks),
                              tickangle=-45, showgrid=True, gridcolor="#eeeeee",
                              range=[global_t_min, global_t_max], row=row, col=1)
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.035), row=4, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", title_text="\u00b5S")
        fig.update_layout(
            title=dict(text=(f"{label} - STL Decomposition (daily period = 96 steps)<br>"
                             "<sup>Annual period not separable: fewer than 2 full cycles available. "
                             "X-axis spans the full monitoring period across all sensors.</sup>"),
                      font=dict(size=12, family=FONT), x=0.0, xanchor="left"),
            height=780, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=65, r=40, t=95, b=110), font=dict(family=FONT),
        )
        save_html(fig, os.path.join(out_dir, f"decomposition_{sensor_key}.html"))

    combined = None
    for sensor_key, dd in all_decomposed.items():
        combined = dd["df"] if combined is None else pd.merge(combined, dd["df"], on="Timestamp", how="outer")
    combined = combined.sort_values("Timestamp").reset_index(drop=True)
    combined.to_csv(safe_path(os.path.join(out_dir, "All_Sensors_Decomposed.csv")),
                     index=False, encoding="utf-8-sig")
    ok(f"All_Sensors_Decomposed.csv saved: {len(combined):,} rows")
    return all_decomposed, combined


# ══════════════════════════════════════════════════════════════════════════
# Step 4b logic: daily seasonal-cycle amplitude vs daily temperature range
# ══════════════════════════════════════════════════════════════════════════

def _exclude_ranges(timestamp_series, exclude_ranges):
    """Boolean mask: True where timestamp_series falls OUTSIDE all excluded ranges."""
    keep = pd.Series(True, index=timestamp_series.index)
    for start_str, end_str in exclude_ranges:
        start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
        keep &= ~((timestamp_series >= start) & (timestamp_series <= end))
    return keep


def run_step4b(all_decomposed, master_for_analysis, out_dir):
    subsection("STEP 4b - Daily seasonal-cycle amplitude vs daily temperature range")

    temp_df = master_for_analysis[["Timestamp", TEMP_COL]].dropna(subset=[TEMP_COL]).copy()
    keep = _exclude_ranges(temp_df["Timestamp"], STEP4B4C_EXCLUDE_RANGES)
    n_excluded = (~keep).sum()
    if n_excluded:
        ok(f"Excluded {n_excluded} temperature rows in confirmed cabling-change window")
    temp_df = temp_df.loc[keep]
    temp_df["Date"] = temp_df["Timestamp"].dt.date
    daily_temp = temp_df.groupby("Date")[TEMP_COL].agg(lambda x: x.max() - x.min())
    daily_temp.name = "Temp_Daily_Range"

    valid_sensors = [(sk, dd) for sk, dd in all_decomposed.items()]
    n_sensors = len(valid_sensors)
    if n_sensors == 0:
        warn("No decomposed sensors available — skipping Step 4b.")
        return

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=n_sensors, cols=1, shared_xaxes=True,
        subplot_titles=[sk.replace("_", " ") for sk, _ in valid_sensors],
        specs=[[{"secondary_y": True}]] * n_sensors,
        vertical_spacing=0.03,
    )

    summary_rows = []
    for i, (sensor_key, dd) in enumerate(valid_sensors, start=1):
        seasonal = dd["seasonal"]
        timestamp = dd["timestamp"]
        colour = dd["colour"]

        s_df = pd.DataFrame({"Timestamp": timestamp.values, "seasonal": seasonal.values}).dropna()
        keep_s = _exclude_ranges(s_df["Timestamp"], STEP4B4C_EXCLUDE_RANGES)
        s_df = s_df.loc[keep_s]
        s_df["Date"] = s_df["Timestamp"].dt.date
        daily_seasonal = s_df.groupby("Date")["seasonal"].agg(lambda x: x.max() - x.min())
        daily_seasonal.name = "Seasonal_Daily_Range"

        merged = pd.concat([daily_seasonal, daily_temp], axis=1).dropna()
        if len(merged) < 3:
            warn(f"{sensor_key}: insufficient overlapping daily data — skipped")
            continue

        r, p = stats.pearsonr(merged["Temp_Daily_Range"], merged["Seasonal_Daily_Range"])
        ok(f"{sensor_key}: r={r:.3f}  p={p:.4g}  n_days={len(merged)}")
        summary_rows.append({"Sensor": sensor_key, "Seasonal_Daily_r": r,
                              "Seasonal_Daily_p": p, "n_days": len(merged)})

        dates = pd.to_datetime(merged.index)
        fig.add_trace(go.Scatter(x=dates, y=merged["Seasonal_Daily_Range"], mode="lines",
                                  line=dict(color=colour, width=1.5),
                                  name=f"{sensor_key} seasonal range", showlegend=False),
                      row=i, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=dates, y=merged["Temp_Daily_Range"], mode="lines",
                                  line=dict(color="#555555", width=1.5, dash="dot"),
                                  name="Temp range", showlegend=(i == 1)),
                      row=i, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Strain range", row=i, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Temp range", row=i, col=1, secondary_y=True)

    fig.update_layout(
        font=dict(family=FONT, size=11), height=280 * n_sensors,
        legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
        margin=dict(l=65, r=60, t=60, b=40), plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    save_html(fig, os.path.join(out_dir, "daily_seasonal_vs_temperature.html"))

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            safe_path(os.path.join(out_dir, "daily_seasonal_vs_temperature.csv")), index=False)
        ok("Saved: daily_seasonal_vs_temperature.csv")


# ══════════════════════════════════════════════════════════════════════════
# Step 4c logic: average diurnal profile by season
# ══════════════════════════════════════════════════════════════════════════

_MONTH_TO_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
_SEASON_COLOURS = {"Winter": "#2196F3", "Spring": "#4CAF50", "Summer": "#FF9800", "Autumn": "#8B4513"}
_SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]


def run_step4c(all_decomposed, out_dir):
    subsection("STEP 4c - Average diurnal profile by season")

    valid_sensors = [(sk, dd) for sk, dd in all_decomposed.items()]
    n_sensors = len(valid_sensors)
    if n_sensors == 0:
        warn("No decomposed sensors available — skipping Step 4c.")
        return

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=n_sensors, cols=1,
                        subplot_titles=[sk.replace("_", " ") for sk, _ in valid_sensors],
                        vertical_spacing=0.05)

    summary_rows = []
    for i, (sensor_key, dd) in enumerate(valid_sensors, start=1):
        seasonal = dd["seasonal"]
        timestamp = dd["timestamp"]

        s_df = pd.DataFrame({"Timestamp": timestamp.values, "seasonal": seasonal.values}).dropna()
        keep_s = _exclude_ranges(s_df["Timestamp"], STEP4B4C_EXCLUDE_RANGES)
        s_df = s_df.loc[keep_s]
        s_df["Season"] = s_df["Timestamp"].dt.month.map(_MONTH_TO_SEASON)
        s_df["TimeOfDay"] = s_df["Timestamp"].dt.strftime("%H:%M")
        s_df["MinutesOfDay"] = s_df["Timestamp"].dt.hour * 60 + s_df["Timestamp"].dt.minute

        for season in _SEASON_ORDER:
            season_data = s_df.loc[s_df["Season"] == season]
            if season_data.empty:
                continue
            profile = season_data.groupby(["MinutesOfDay", "TimeOfDay"])["seasonal"].mean().reset_index()
            profile = profile.sort_values("MinutesOfDay")
            n_days = season_data["Timestamp"].dt.date.nunique()
            ok(f"{sensor_key} / {season}: n_days={n_days}")

            fig.add_trace(go.Scatter(x=profile["TimeOfDay"], y=profile["seasonal"], mode="lines",
                                      line=dict(color=_SEASON_COLOURS[season], width=2),
                                      name=season, showlegend=(i == 1), legendgroup=season),
                          row=i, col=1)

            peak_idx = profile["seasonal"].idxmax()
            trough_idx = profile["seasonal"].idxmin()
            summary_rows.append({
                "Sensor": sensor_key, "Season": season,
                "Peak_Time": profile.loc[peak_idx, "TimeOfDay"], "Peak_Value": profile.loc[peak_idx, "seasonal"],
                "Trough_Time": profile.loc[trough_idx, "TimeOfDay"], "Trough_Value": profile.loc[trough_idx, "seasonal"],
                "Amplitude": profile.loc[peak_idx, "seasonal"] - profile.loc[trough_idx, "seasonal"],
                "N_Days": n_days,
            })

        fig.update_yaxes(title_text="Strain (seasonal daily)", row=i, col=1)
        if i == n_sensors:
            fig.update_xaxes(title_text="Time of day", row=i, col=1, tickmode="array",
                              tickvals=["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"])
        else:
            fig.update_xaxes(showticklabels=False, row=i, col=1)

    fig.update_layout(
        font=dict(family=FONT, size=11), height=280 * n_sensors,
        legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
        margin=dict(l=65, r=40, t=60, b=60), plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    save_html(fig, os.path.join(out_dir, "diurnal_profile_by_season.html"))

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            safe_path(os.path.join(out_dir, "diurnal_profile_by_season.csv")), index=False)
        ok("Saved: diurnal_profile_by_season.csv")


# ══════════════════════════════════════════════════════════════════════════
# Step 5 logic: trend analysis (combined + bursts + recorded events)
# ══════════════════════════════════════════════════════════════════════════

def run_step5(all_decomposed, raw_burst_data, out_dir):
    subsection("STEP 5 - Trend analysis (combined, plus bursts and recorded events)")
    if not all_decomposed:
        warn("No decomposed data available — Step 5 skipped")
        return

    t_min = min(dd["timestamp"].min() for dd in all_decomposed.values())
    t_max = max(dd["timestamp"].max() for dd in all_decomposed.values())

    fig1 = plotly_base("All Sensors - STL Trend Components (Combined)",
                       "Temperature-compensated strain trend, all sensors on one chart.", height=680)
    fig1.update_yaxes(title_text="Trend (\u00b5S)")
    for sensor_key, dd in all_decomposed.items():
        label = sensor_key.replace("_", " ")
        fig1.add_trace(go.Scatter(x=dd["timestamp"], y=dd["trend"], mode="lines", name=label,
                                   line=dict(color=dd["colour"], width=1.0), connectgaps=False,
                                   showlegend=True,
                                   hovertemplate=f"{label}<br>%{{x|%d %b %Y}}<br>%{{y:.3f}} \u00b5S<extra></extra>"))
    fig1.update_layout(legend=dict(orientation="h", y=-0.55, x=0, xanchor="left", font=dict(size=11),
                                   bgcolor="rgba(255,255,255,0.95)", bordercolor="#cccccc", borderwidth=1))
    fig1.update_layout(margin=dict(l=65, r=60, t=90, b=260))
    apply_month_ticks(fig1, t_min, t_max)
    save_html(fig1, os.path.join(out_dir, "trend_all_sensors.html"))

    fig2 = plotly_base("All Sensors - STL Trend Components with Burst Events and Recorded Events",
                       "Bold markers = burst event (per sensor, placed on that sensor's own trend value). "
                       "Shaded/vertical lines = recorded site events, coloured by category.", height=900)
    fig2.update_yaxes(title_text="Trend (\u00b5S)")
    for sensor_key, dd in all_decomposed.items():
        label = sensor_key.replace("_", " ")
        fig2.add_trace(go.Scatter(x=dd["timestamp"], y=dd["trend"], mode="lines", name=label,
                                   line=dict(color=dd["colour"], width=1.0), connectgaps=False,
                                   legendgroup=label,
                                   hovertemplate=f"{label}<br>%{{x|%d %b %Y}}<br>%{{y:.3f}} \u00b5S<extra></extra>"))

    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        if sensor_key not in all_decomposed or sensor_key not in raw_burst_data:
            continue
        bdf = raw_burst_data[sensor_key]
        burst_col = f"{sensor_key}_Burst(\u00b5S)"
        if bdf.empty or burst_col not in bdf.columns:
            continue
        cutoff = pd.Timestamp(SETTLING_CUTOFF[folder])
        bdf = bdf[bdf["Timestamp"] >= cutoff]
        if bdf.empty:
            continue
        events = summarize_burst_events(bdf, "Timestamp", burst_col)
        if events.empty:
            continue
        trend_df = pd.DataFrame({"Timestamp": all_decomposed[sensor_key]["timestamp"],
                                  "trend": all_decomposed[sensor_key]["trend"].values})
        merged = pd.merge_asof(events.sort_values("Timestamp"), trend_df.sort_values("Timestamp"),
                               on="Timestamp", direction="nearest", tolerance=pd.Timedelta(hours=6)).dropna(subset=["trend"])
        if merged.empty:
            continue
        label = sensor_key.replace("_", " ")
        fig2.add_trace(go.Scatter(x=merged["Timestamp"], y=merged["trend"], mode="markers",
                                   name=f"{label} (burst)", marker=dict(color=colour, size=9,
                                   line=dict(color="black", width=1)), legendgroup=label, showlegend=False,
                                   hovertemplate=f"{label} burst event<br>%{{x|%d %b %Y %H:%M:%S}}<extra></extra>"))

    # ── Recorded events: shaded spans for multi-day events, dashed
    #    vertical lines for everything else ───────────────────────────
    # A multi-day event is any category that appears MORE THAN ONCE
    # within the same block (a run of dated rows with no blank row
    # between them, per the sheet's own convention) — the span runs
    # from the earliest to the latest date of that category within the
    # block. A category appearing only once in a block is a single-day
    # event and gets a dashed line instead. Derived directly from the
    # sheet's structure each run — nothing to maintain by hand.
    events = load_recorded_events(EVENTS_FILE)
    groups = {}
    for ev in events:
        groups.setdefault((ev["block_id"], ev["category"]), []).append(ev)

    consumed_ids = set()
    for (block_id, category), members in groups.items():
        if len(members) < 2:
            continue   # single occurrence in this block — dashed line, not a span
        start = min(m["date"] for m in members)
        end = max(m["date"] for m in members)
        colour = EVENT_CATEGORY_COLOURS[category]
        fig2.add_vrect(
            x0=start, x1=end + pd.Timedelta(days=1),
            fillcolor=colour, opacity=0.18, layer="below", line_width=0,
        )
        consumed_ids.update(id(m) for m in members)

    seen_categories = set()
    for ev in events:
        if ev["date"] < t_min or ev["date"] > t_max:
            continue
        if id(ev) in consumed_ids:
            # Already represented by a shaded span above.
            seen_categories.add(ev["category"])
            continue
        colour = EVENT_CATEGORY_COLOURS[ev["category"]]
        seen_categories.add(ev["category"])
        fig2.add_vline(
            x=ev["date"].timestamp() * 1000,
            line_width=2.5, line_dash="dash", line_color=colour, opacity=0.85,
        )
    # Legend entries for the three event categories (one dummy trace each)
    category_labels = {"weather": "Weather event", "around_aircraft": "Event around the Canberra",
                       "interference": "Stepping on / interference"}
    for cat, colour in EVENT_CATEGORY_COLOURS.items():
        if cat in seen_categories:
            fig2.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                       line=dict(color=colour, width=2.5, dash="dash"),
                                       name=category_labels[cat]))
    apply_month_ticks(fig2, t_min, t_max)
    fig2.update_layout(legend=dict(orientation="h", y=-0.37, x=0, xanchor="left", font=dict(size=11),
                                   bgcolor="rgba(255,255,255,0.95)", bordercolor="#cccccc", borderwidth=1))
    fig2.update_layout(margin=dict(l=65, r=60, t=90, b=320))
    save_html(fig2, os.path.join(out_dir, "trend_with_bursts_and_events.html"))


# ══════════════════════════════════════════════════════════════════════════
# Step 6a/6b logic: correlation (compensated strain / trend)
# ══════════════════════════════════════════════════════════════════════════

def run_correlation(data_by_sensor, col_lookup, out_dir, source_label):
    labels, sk_list, col_list = [], [], []
    for sensor_key in SENSOR_MAP:
        sensor_full = SENSOR_MAP[sensor_key][0]
        col = col_lookup(sensor_key)
        if col is not None:
            labels.append(sensor_full.replace("_", " "))
            sk_list.append(sensor_key)
            col_list.append(col)
    n_sensors = len(col_list)
    if n_sensors < 2:
        warn(f"Fewer than 2 sensors available for {source_label} correlation — skipped")
        return

    r_matrix = np.full((n_sensors, n_sensors), np.nan)
    p_matrix = np.full((n_sensors, n_sensors), np.nan)
    corr_rows = []
    for i in range(n_sensors):
        for j in range(n_sensors):
            if i <= j:
                continue
            mask = data_by_sensor[col_list[i]].notna() & data_by_sensor[col_list[j]].notna()
            n_pair = int(mask.sum())
            if n_pair < 3:
                continue
            si = data_by_sensor.loc[mask, col_list[i]].to_numpy()
            sj = data_by_sensor.loc[mask, col_list[j]].to_numpy()
            r_val, p_val = stats.pearsonr(si, sj)
            r_matrix[i, j] = r_val
            p_matrix[i, j] = p_val
            corr_rows.append({"Sensor A": labels[i], "Sensor B": labels[j],
                              "Pearson r": round(float(r_val), 4), "p-value": float(p_val),
                              "Significance": sig_stars(p_val), "n": n_pair})
    ok(f"{source_label}: {len(corr_rows)} pair(s) computed")
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(safe_path(os.path.join(out_dir, "correlation_table.csv")),
                    index=False, encoding="utf-8-sig")

    annotations, hover_text = [], []
    for i in range(n_sensors):
        row_hover = []
        for j in range(n_sensors):
            if i <= j:
                row_hover.append("")
                continue
            r_val, p_val = r_matrix[i, j], p_matrix[i, j]
            if np.isnan(r_val):
                row_hover.append("")
                continue
            stars = sig_stars(p_val)
            p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
            dark_cell = abs(r_val) > 0.6
            text_colour = "white" if dark_cell else "#333333"
            annotations.append(dict(x=labels[j], y=labels[i],
                                    text=f"<b>{r_val:.2f}{stars}</b><br><span style='font-size:8px'>{p_text}</span>",
                                    font=dict(size=11, family=FONT, color=text_colour),
                                    showarrow=False, align="center"))
            n_pair = int((data_by_sensor[col_list[i]].notna() & data_by_sensor[col_list[j]].notna()).sum())
            p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"
            row_hover.append(f"<b>{labels[i]}</b> vs <b>{labels[j]}</b><br>r = {r_val:.4f}{stars}<br>p {p_str}<br>n = {n_pair:,}")
        hover_text.append(row_hover)

    fig_heat = go.Figure(go.Heatmap(
        z=r_matrix, x=labels, y=labels, text=hover_text, hovertemplate="%{text}<extra></extra>",
        colorscale=[[0.0, "#1565C0"], [0.25, "#90CAF9"], [0.5, "#FFFFFF"], [0.75, "#EF9A9A"], [1.0, "#B71C1C"]],
        zmin=-1, zmax=1, colorbar=dict(title=dict(text="Pearson r", side="right"),
                                       tickvals=[-1, -0.5, 0, 0.5, 1], thickness=16),
    ))
    fig_heat.update_layout(
        title=dict(text=(f"Inter-Sensor Pearson Correlation ({source_label}) - Lower Triangle Heatmap<br>"
                         "<sup>Significance: *** p&lt;0.001  ** p&lt;0.01  * p&lt;0.05.</sup>"),
                  font=dict(size=13, family=FONT), x=0.0, xanchor="left"),
        annotations=annotations, xaxis=dict(tickfont=dict(size=10), side="bottom"),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white", height=600, width=720,
        margin=dict(l=150, r=80, t=110, b=130), font=dict(family=FONT),
    )
    save_html(fig_heat, os.path.join(out_dir, "correlation_heatmap.html"))

    from itertools import combinations
    rolling_steps = ROLLING_DAYS * STEPS_PER_DAY
    pairs = list(combinations(range(n_sensors), 2))
    fig_roll = go.Figure()
    fig_roll.update_layout(
        title=dict(text=(f"Rolling Pearson Correlation ({ROLLING_DAYS}-day window, {source_label}) - "
                         "All Sensor Pairs"), font=dict(size=13, family=FONT), x=0.0, xanchor="left"),
        xaxis=dict(showgrid=True, gridcolor="#eeeeee", type="date", rangeslider=dict(visible=True, thickness=0.035)),
        yaxis=dict(title="Pearson r", range=[-1.05, 1.05],
                  tickvals=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], showgrid=True, gridcolor="#eeeeee"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.55, x=0, xanchor="left", font=dict(size=10),
                   bgcolor="rgba(255,255,255,0.95)", bordercolor="#cccccc", borderwidth=1, tracegroupgap=4),
        height=820, margin=dict(l=65, r=60, t=100, b=320), hovermode="x unified", font=dict(family=FONT),
    )
    fig_roll.add_hline(y=0, line_dash="dot", line_color="#bbbbbb", line_width=1)
    fig_roll.add_hline(y=0.7, line_dash="dash", line_color="#d62728", line_width=0.8,
                       annotation_text="r = 0.7", annotation_position="top right",
                       annotation_font_size=9, annotation_font_color="#d62728")
    fig_roll.add_hline(y=-0.7, line_dash="dash", line_color="#1f77b4", line_width=0.8,
                       annotation_text="r = -0.7", annotation_position="bottom right",
                       annotation_font_size=9, annotation_font_color="#1f77b4")
    for k, (i, j) in enumerate(pairs):
        pair_name = f"{labels[i]} / {labels[j]}"
        si = data_by_sensor[col_list[i]].reset_index(drop=True)
        sj = data_by_sensor[col_list[j]].reset_index(drop=True)
        roll_r = si.rolling(window=rolling_steps, min_periods=rolling_steps // 2).corr(sj)
        fig_roll.add_trace(go.Scatter(x=data_by_sensor["Timestamp"], y=roll_r.values, mode="lines",
                                       name=pair_name, line=dict(color=PAIR_COLOURS[k % len(PAIR_COLOURS)], width=1.0),
                                       connectgaps=False,
                                       hovertemplate=f"<b>{pair_name}</b><br>%{{x|%d %b %Y}}<br>r = %{{y:.3f}}<extra></extra>"))
    t_min, t_max = data_by_sensor["Timestamp"].min(), data_by_sensor["Timestamp"].max()
    apply_month_ticks(fig_roll, t_min, t_max)
    save_html(fig_roll, os.path.join(out_dir, "correlation_rolling.html"))


# ══════════════════════════════════════════════════════════════════════════
# Step 7 logic: trend model fitting
# ══════════════════════════════════════════════════════════════════════════

def compute_aic_bic(n, k, rss):
    if n <= k or rss <= 0:
        return np.inf, np.inf
    sigma2 = rss / n
    log_lik = -n / 2 * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
    return 2 * k - 2 * log_lik, k * np.log(n) - 2 * log_lik


def damped_sinusoid(t, A, omega, phi, gamma, C):
    return A * np.exp(-gamma * t) * np.sin(omega * t + phi) + C


def run_step7(compensated_data, out_dir):
    subsection("STEP 7 - Trend model fitting (daily median, AIC/BIC selection)")
    all_model_rows = []

    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        if sensor_key not in compensated_data:
            continue
        d = compensated_data[sensor_key]
        df = d["df"].copy()
        comp_col = d["comp_col"]
        label = sensor_key.replace("_", " ")
        df["Date"] = df["Timestamp"].dt.normalize()
        daily = df.groupby("Date")[comp_col].median().reset_index()
        daily.columns = ["Date", "y"]
        daily = daily.dropna()
        if len(daily) < 10:
            warn(f"{sensor_key}: insufficient daily rows — skipped")
            continue

        t0 = daily["Date"].min()
        t = ((daily["Date"] - t0).dt.total_seconds() / 86400).to_numpy()
        y = daily["y"].to_numpy()
        n = len(y)
        candidates = []

        for deg in range(1, 6):
            X = np.column_stack([t ** dd for dd in range(deg + 1)])
            k = deg + 1
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                y_hat = X @ coeffs
                rss = float(np.sum((y - y_hat) ** 2))
                aic, bic = compute_aic_bic(n, k, rss)
                r2 = 1 - rss / np.sum((y - y.mean()) ** 2)
                candidates.append(dict(name=f"Polynomial deg {deg}", k=k, aic=aic, bic=bic, r2=r2, y_hat=y_hat))
            except Exception:
                pass

        try:
            p0 = [y.std(), 2 * np.pi / 365.0, 0, 1e-4, y.mean()]
            popt, _ = curve_fit(damped_sinusoid, t, y, p0=p0, maxfev=10000,
                                bounds=([-np.inf, 1e-6, -np.pi, 0, -np.inf], [np.inf, np.inf, np.pi, np.inf, np.inf]))
            y_hat = damped_sinusoid(t, *popt)
            rss = float(np.sum((y - y_hat) ** 2))
            aic, bic = compute_aic_bic(n, 5, rss)
            r2 = 1 - rss / np.sum((y - y.mean()) ** 2)
            candidates.append(dict(name="Damped sinusoid", k=5, aic=aic, bic=bic, r2=r2, y_hat=y_hat))
        except Exception as e:
            warn(f"{sensor_key}: damped sinusoid fit failed ({e})")

        try:
            algo = ruptures.Dynp(model="l2").fit(y.reshape(-1, 1))
            bkpts = algo.predict(n_bkps=1)
            bp = bkpts[0]
            X_pw = np.zeros((n, 4))
            X_pw[:, 0] = 1
            X_pw[:, 1] = t
            X_pw[:, 2] = (t > t[bp]).astype(float) if bp < n else 0
            X_pw[:, 3] = np.where(t > t[bp], t - t[bp], 0) if bp < n else 0
            coeffs_pw = np.linalg.lstsq(X_pw, y, rcond=None)[0]
            y_hat = X_pw @ coeffs_pw
            rss = float(np.sum((y - y_hat) ** 2))
            aic, bic = compute_aic_bic(n, 4, rss)
            r2 = 1 - rss / np.sum((y - y.mean()) ** 2)
            candidates.append(dict(name="Piecewise linear", k=4, aic=aic, bic=bic, r2=r2, y_hat=y_hat,
                                   bp_date=daily["Date"].iloc[min(bp, n - 1)]))
        except Exception as e:
            warn(f"{sensor_key}: piecewise linear fit failed ({e})")

        if not candidates:
            warn(f"{sensor_key}: no models fitted — skipped")
            continue

        try:
            lowess_y = lowess(y, t, frac=0.3, return_sorted=True)[:, 1]
        except Exception:
            lowess_y = None

        best_aic = min(candidates, key=lambda c: c["aic"])
        best_bic = min(candidates, key=lambda c: c["bic"])
        if best_aic["name"] == best_bic["name"]:
            best, sel_note = best_aic, "AIC and BIC agree"
        else:
            best = best_bic
            sel_note = f"AIC->{best_aic['name']}, BIC->{best_bic['name']} — BIC (simpler) selected"
        ok(f"{sensor_key}: best={best['name']}  AIC={best['aic']:.1f}  BIC={best['bic']:.1f}  R2={best['r2']:.4f}")

        for c in candidates:
            all_model_rows.append({"Sensor": label, "Model": c["name"], "k": c["k"],
                                    "AIC": round(c["aic"], 2), "BIC": round(c["bic"], 2), "R2": round(c["r2"], 4),
                                    "Selected": "YES" if c["name"] == best["name"] else "",
                                    "Note": sel_note if c["name"] == best["name"] else ""})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["Date"], y=y, mode="markers",
                                  marker=dict(color=colour, size=3, opacity=0.5), name="Daily median",
                                  hovertemplate="%{x|%d %b %Y}<br>%{y:.3f} \u00b5S<extra></extra>"))
        colour_idx = 0
        for c in candidates:
            if c["name"] == best["name"]:
                continue
            fig.add_trace(go.Scatter(x=daily["Date"], y=c["y_hat"], mode="lines",
                                     name=f"{c['name']}  (AIC={c['aic']:.0f})",
                                     line=dict(color=NON_BEST_COLOURS[colour_idx % len(NON_BEST_COLOURS)], width=1.0, dash="dot"),
                                     opacity=0.55,
                                     hovertemplate=f"{c['name']}<br>%{{x|%d %b %Y}}<br>%{{y:.3f}} \u00b5S<extra></extra>"))
            colour_idx += 1
        if lowess_y is not None:
            fig.add_trace(go.Scatter(x=daily["Date"], y=lowess_y, mode="lines", name="LOWESS (visual reference)",
                                     line=dict(color="#607D8B", width=1.2, dash="dot"), opacity=0.7))
        fig.add_trace(go.Scatter(x=daily["Date"], y=best["y_hat"], mode="lines",
                                 name=f"\u2605 Best: {best['name']}  (AIC={best['aic']:.0f}, R\u00b2={best['r2']:.4f})",
                                 line=dict(color="#111111", width=2.2),
                                 hovertemplate=f"Best fit: {best['name']}<br>%{{x|%d %b %Y}}<br>%{{y:.3f}} \u00b5S<extra></extra>"))
        if "bp_date" in best:
            fig.add_vline(x=best["bp_date"].timestamp() * 1000, line_dash="dot", line_color="#999999",
                         annotation_text="Breakpoint", annotation_font_size=10, annotation_position="top right")
        fig.update_layout(
            title=dict(text=(f"{label} - Trend Model Fitting (daily median)<br>"
                             f"<sup>\u2605 Best: {best['name']}  |  R\u00b2={best['r2']:.4f}  |  "
                             f"AIC={best['aic']:.1f}  |  BIC={best['bic']:.1f}  |  {sel_note}</sup>"),
                      font=dict(size=12, family=FONT), x=0.0, xanchor="left"),
            xaxis=dict(title="Date", showgrid=True, gridcolor="#eeeeee",
                      rangeslider=dict(visible=True, thickness=0.04), type="date"),
            yaxis=dict(title="Compensated Strain (\u00b5S)", showgrid=True, gridcolor="#eeeeee"),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.38, x=0, xanchor="left", font=dict(size=10),
                       bgcolor="rgba(255,255,255,0.95)", bordercolor="#cccccc", borderwidth=1),
            height=680, margin=dict(l=65, r=50, t=100, b=190), font=dict(family=FONT),
        )
        save_html(fig, os.path.join(out_dir, f"model_fit_{sensor_key}.html"))

    if all_model_rows:
        pd.DataFrame(all_model_rows).to_csv(safe_path(os.path.join(out_dir, "aic_bic_summary.csv")),
                                             index=False, encoding="utf-8-sig")
        ok("aic_bic_summary.csv saved")


# ══════════════════════════════════════════════════════════════════════════
# Round / seed management + Step 1 (incremental)
# ══════════════════════════════════════════════════════════════════════════

def get_step_dir(round_dir, name):
    d = os.path.join(round_dir, name)
    os.makedirs(safe_path(d), exist_ok=True)
    return d


def load_seed_meta(path):
    """
    Loads the persisted per-series processing watermark (Series ->
    last-processed Timestamp). This is DELIBERATELY separate from
    scanning the seed file's own data for the "last non-null" value —
    a few raw readings right at the tail of a sensor's data can fail to
    align within tolerance and land as NaN in the grid; using "last
    non-null" as the watermark would make those same tail readings look
    "new" again on every subsequent run (an infinite false-positive
    loop). The watermark instead reflects the true extent of what was
    already processed (the aligned grid's actual max timestamp),
    regardless of whether that specific point got a value.
    """
    if not os.path.isfile(safe_path(path)):
        return {}
    meta = pd.read_csv(safe_path(path), parse_dates=["LastProcessedTimestamp"])
    return dict(zip(meta["Series"], meta["LastProcessedTimestamp"]))


def save_seed_meta(path, watermarks):
    rows = [{"Series": k, "LastProcessedTimestamp": v} for k, v in watermarks.items() if v is not None]
    pd.DataFrame(rows).to_csv(safe_path(path), index=False, encoding="utf-8-sig")


def main():
    run_started = datetime.now()
    section("CANBERRA PR9 SHM - FULL PIPELINE (incremental)")
    print(f"  Run started: {run_started.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Excel engine: {EXCEL_ENGINE}")
    print(f"  Base directory: {BASE_DIR}")

    seed_path = os.path.join(BASE_DIR, SEED_FILE)
    seed_meta_path = os.path.join(BASE_DIR, SEED_META_FILE)
    seed_exists = os.path.isfile(safe_path(seed_path))
    if seed_exists:
        seed = pd.read_csv(safe_path(seed_path), parse_dates=["Timestamp"])
        ok(f"Existing seed loaded: {len(seed):,} rows")
    else:
        seed = None
        ok("No existing seed found — this run will be treated as the first-ever round.")

    watermarks = load_seed_meta(seed_meta_path)
    seed_last = {}
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        seed_last[sensor_key] = watermarks.get(sensor_key)
    temp_seed_last = watermarks.get("Exhibition_Temp")

    section("Reading raw sensor data")
    raw_data = {}
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        sensor_path = os.path.join(BASE_DIR, SENSORS_SUBDIR, folder)
        if not os.path.isdir(sensor_path):
            warn(f"Sensor folder not found, skipped: {sensor_path}")
            raw_data[sensor_key] = (pd.DataFrame(columns=["Timestamp", f"{sensor_key}_Strain(\u00b5S)"]),
                                     pd.DataFrame(columns=["Timestamp", f"{sensor_key}_Burst(\u00b5S)"]))
            continue
        raw_data[sensor_key] = load_sensor_files(sensor_path, sensor_key)

    temp_df = load_exhibition_temp(os.path.join(BASE_DIR, EXHIBITION_SUBDIR))

    section("Checking for new data")
    new_data_info = {}
    any_new = False
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        raw_strain, _ = raw_data[sensor_key]
        last_ts = seed_last[sensor_key]
        new_strain = raw_strain if last_ts is None else raw_strain[raw_strain["Timestamp"] > last_ts]
        new_data_info[sensor_key] = new_strain
        if not new_strain.empty:
            any_new = True
            ok(f"{sensor_key}: {len(new_strain):,} new row(s) since "
               f"{'(no prior seed)' if last_ts is None else last_ts}")
        else:
            print(f"      {sensor_key}: no new data")

    new_temp = temp_df if temp_seed_last is None else temp_df[temp_df["Timestamp"] > temp_seed_last]
    if not new_temp.empty:
        any_new = True
        ok(f"Exhibition temperature: {len(new_temp):,} new row(s)")
    else:
        print("      Exhibition temperature: no new data")

    if not any_new:
        section("NO NEW DATA")
        print("  No new data found since the last round for any sensor or temperature.")
        print("  Nothing to do — no folder created, no files changed.")
        return

    output_root = os.path.join(BASE_DIR, OUTPUT_BASE)
    os.makedirs(safe_path(output_root), exist_ok=True)
    existing_rounds = [
        d for d in os.listdir(safe_path(output_root))
        if d.startswith("Round ") and os.path.isdir(os.path.join(output_root, d))
    ]
    round_num = len(existing_rounds) + 1
    round_name = f"Round {round_num}"
    round_dir = os.path.join(output_root, round_name)
    os.makedirs(safe_path(round_dir), exist_ok=True)
    section(f"NEW DATA FOUND -> starting {round_name}")

    # ── Continuity check (gap / overlap) — warning-only ─────────────────
    continuity_warnings = []
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        last_ts = seed_last[sensor_key]
        new_strain = new_data_info[sensor_key]
        if last_ts is not None and not new_strain.empty:
            gap = new_strain["Timestamp"].min() - last_ts
            if gap > pd.Timedelta(hours=CONTINUITY_GAP_HOURS):
                msg = (f"{sensor_key}: possible GAP — seed's last reading {last_ts}, "
                       f"new data begins {new_strain['Timestamp'].min()} (gap = {gap})")
                continuity_warnings.append(msg)
                warn(msg)
        if seed is not None and last_ts is not None:
            raw_strain, _ = raw_data[sensor_key]
            tail_raw = raw_strain[raw_strain["Timestamp"] >= last_ts - pd.Timedelta(days=2)]
            if not tail_raw.empty:
                tail_aligned = align_to_15min(tail_raw, "Timestamp", strain_col)
                overlap_region = tail_aligned[tail_aligned["Timestamp"] <= last_ts]
                seed_tail = seed[["Timestamp", strain_col]].dropna() if strain_col in seed.columns else pd.DataFrame()
                if not overlap_region.empty and not seed_tail.empty:
                    merged_check = overlap_region.merge(seed_tail, on="Timestamp", suffixes=("_new", "_seed"))
                    if not merged_check.empty:
                        diffs = (merged_check[f"{strain_col}_new"] - merged_check[f"{strain_col}_seed"]).abs()
                        bad = diffs[diffs > CONTINUITY_OVERLAP_TOL_UE]
                        if len(bad) > 0:
                            msg = (f"{sensor_key}: possible OVERLAP discrepancy — {len(bad)} timestamp(s) "
                                   f"in the overlapping period differ from the seed by more than "
                                   f"{CONTINUITY_OVERLAP_TOL_UE} \u00b5S (max diff {diffs.max():.3f})")
                            continuity_warnings.append(msg)
                            warn(msg)
    if temp_seed_last is not None and not new_temp.empty:
        gap = new_temp["Timestamp"].min() - temp_seed_last
        if gap > pd.Timedelta(hours=CONTINUITY_GAP_HOURS):
            msg = (f"Exhibition temperature: possible GAP — seed's last reading {temp_seed_last}, "
                   f"new data begins {new_temp['Timestamp'].min()} (gap = {gap})")
            continuity_warnings.append(msg)
            warn(msg)
    if not continuity_warnings:
        ok("Continuity check: no unexpected gaps or overlaps detected")

    # ── Build updated aligned+filled+trimmed series per sensor ──────────
    section("STEP 1 - Alignment, gap-fill, settling trim (incremental)")
    step1_dir = get_step_dir(round_dir, "Step 1 Collation Alignment and Raw Plots")
    updated_sensor_series = {}
    rows_added_by_sensor = {}
    new_watermarks = {}

    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        new_strain = new_data_info[sensor_key]
        last_ts = seed_last[sensor_key]

        if not new_strain.empty:
            if last_ts is not None:
                grid_start = (last_ts + pd.Timedelta(minutes=15)).floor("15min")
                grid_end = new_strain["Timestamp"].max().ceil("15min")
                aligned_new = align_to_15min(new_strain, "Timestamp", strain_col,
                                             grid_start=grid_start, grid_end=grid_end)
            else:
                aligned_new = align_to_15min(new_strain, "Timestamp", strain_col)
            if seed is not None and strain_col in seed.columns:
                combined = pd.concat([seed[["Timestamp", strain_col]], aligned_new], ignore_index=True)
            else:
                combined = aligned_new
            combined = combined.drop_duplicates(subset="Timestamp").sort_values("Timestamp").reset_index(drop=True)
            # "Rows added" = the new 15-min grid points this round actually
            # contributed. By construction (grid_start = last_ts + 15min)
            # these timestamps didn't already exist in the seed, so this is
            # unambiguous — unlike a before/after total-row-count diff,
            # which would be thrown off by the settling trim and by the
            # seed's Timestamp column now spanning temperature's full
            # (much longer) date range rather than just this sensor's own.
            rows_added_by_sensor[sensor_key] = len(aligned_new)
            combined[strain_col] = gap_fill(combined[strain_col])
            cutoff = SETTLING_CUTOFF[folder]
            combined = apply_settling_trim(combined, "Timestamp", cutoff)
        else:
            if seed is not None and strain_col in seed.columns:
                combined = seed[["Timestamp", strain_col]].copy()
                combined = apply_settling_trim(combined, "Timestamp", SETTLING_CUTOFF[folder])
            else:
                combined = pd.DataFrame(columns=["Timestamp", strain_col])
            rows_added_by_sensor[sensor_key] = 0

        updated_sensor_series[sensor_key] = combined
        # Watermark: for a sensor WITH new data this round, use its own
        # freshly-built grid's true extent. For a sensor with NO new data,
        # explicitly keep its own previous watermark rather than deriving
        # one from `combined` — `combined` here is just a column pulled
        # out of the shared multi-sensor seed file, whose Timestamp range
        # reflects the UNION of all sensors, not this one's own true
        # extent. Recomputing from it would incorrectly inflate an
        # unchanged sensor's watermark to match whichever other sensor
        # went furthest.
        if not new_strain.empty:
            new_watermarks[sensor_key] = aligned_new["Timestamp"].max() if not aligned_new.empty else last_ts
        else:
            new_watermarks[sensor_key] = last_ts
        combined.to_csv(safe_path(os.path.join(step1_dir, f"{sensor_key}_aligned_15min_filled_trimmed.csv")),
                        index=False, encoding="utf-8-sig")
        if not new_strain.empty:
            new_strain.to_csv(safe_path(os.path.join(step1_dir, f"{sensor_key}_raw_strain_new_this_round.csv")),
                              index=False, encoding="utf-8-sig")
        ok(f"{sensor_key}: {len(combined):,} total aligned rows "
           f"(+{rows_added_by_sensor[sensor_key]:,} this round)")

    # Temperature: align new rows, append, no gap-fill (matches Step 1 convention)
    if not new_temp.empty:
        if temp_seed_last is not None:
            grid_start = (temp_seed_last + pd.Timedelta(minutes=15)).floor("15min")
            grid_end = new_temp["Timestamp"].max().ceil("15min")
            temp_aligned_new = align_to_15min(new_temp, "Timestamp", TEMP_COL,
                                              grid_start=grid_start, grid_end=grid_end)
        else:
            temp_aligned_new = align_to_15min(new_temp, "Timestamp", TEMP_COL)
        if seed is not None and TEMP_COL in seed.columns:
            temp_combined = pd.concat([seed[["Timestamp", TEMP_COL]], temp_aligned_new], ignore_index=True)
        else:
            temp_combined = temp_aligned_new
        temp_combined = temp_combined.drop_duplicates(subset="Timestamp").sort_values("Timestamp").reset_index(drop=True)
    else:
        temp_combined = seed[["Timestamp", TEMP_COL]].copy() if (seed is not None and TEMP_COL in seed.columns) else pd.DataFrame(columns=["Timestamp", TEMP_COL])
    temp_combined.to_csv(safe_path(os.path.join(step1_dir, "Exhibition_Temp_aligned_15min.csv")),
                         index=False, encoding="utf-8-sig")
    new_watermarks["Exhibition_Temp"] = (
        temp_aligned_new["Timestamp"].max() if not new_temp.empty and not temp_aligned_new.empty else temp_seed_last
    )

    # ── Build and save the new seed ──────────────────────────────────────
    new_seed = None
    for sensor_key, combined in updated_sensor_series.items():
        new_seed = combined if new_seed is None else pd.merge(new_seed, combined, on="Timestamp", how="outer")
    new_seed = pd.merge(new_seed, temp_combined, on="Timestamp", how="outer")
    new_seed = new_seed.sort_values("Timestamp").reset_index(drop=True)
    new_seed.to_csv(safe_path(seed_path), index=False, encoding="utf-8-sig")
    save_seed_meta(seed_meta_path, new_watermarks)
    ok(f"Seed metadata (processing watermarks) saved to {SEED_META_FILE}")
    total_rows_added = len(new_seed) - (len(seed) if seed is not None else 0)
    ok(f"Seed updated and overwritten: {len(new_seed):,} total rows (+{total_rows_added:,} this round)")

    new_seed.to_csv(safe_path(os.path.join(step1_dir, "All_Sensors_Aligned_Filled_Trimmed.csv")),
                    index=False, encoding="utf-8-sig")

    # ── Full burst re-collation (cheap, not incremental — see docstring) ─
    raw_burst_data = {}
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        _, raw_burst = raw_data[sensor_key]
        raw_burst_data[sensor_key] = raw_burst
        raw_burst.to_csv(safe_path(os.path.join(step1_dir, f"{sensor_key}_raw_burst.csv")),
                         index=False, encoding="utf-8-sig")

    # ── Step 1 plots (same as standalone script, on the FULL new seed) ──
    t_min = min(updated_sensor_series[sk]["Timestamp"].min() for sk in updated_sensor_series
                if not updated_sensor_series[sk].empty)
    t_max = max(updated_sensor_series[sk]["Timestamp"].max() for sk in updated_sensor_series
                if not updated_sensor_series[sk].empty)
    global_cutoff = pd.Timestamp(min(SETTLING_CUTOFF.values()))
    temp_plot = temp_combined[temp_combined["Timestamp"] >= global_cutoff]

    fig_a = plotly_base("All Sensors - Strain (15-minute aligned, gap-filled, settling-trimmed)",
                        "Pre-compensation strain, per sensor. Dashed = gap in the record.")
    fig_a.update_yaxes(title_text="Strain (\u00b5S)")
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        df = updated_sensor_series[sensor_key]
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        if df.empty:
            continue
        label = sensor_key.replace("_", " ")
        add_gapped_line(fig_a, df["Timestamp"], df[strain_col], label, colour,
                        hovertemplate=f"{label}<br>%{{x|%d %b %Y %H:%M}}<br>%{{y:.3f}} \u00b5S<extra></extra>")
    apply_month_ticks(fig_a, t_min, t_max)
    save_html(fig_a, os.path.join(step1_dir, "strain_all_sensors_15min.html"))

    fig_b = plotly_base("All Sensors - Strain with Exhibition Temperature (15-minute aligned)",
                        "Strain on left axis (per sensor), exhibition temperature on right axis. Dashed = gap in the record.")
    fig_b.update_layout(yaxis=dict(title="Strain (\u00b5S)", showgrid=True, gridcolor="#eeeeee"),
                        yaxis2=dict(title="Exhibition Temp (\u00b0C)", overlaying="y", side="right", showgrid=False))
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        df = updated_sensor_series[sensor_key]
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        if df.empty:
            continue
        label = sensor_key.replace("_", " ")
        add_gapped_line(fig_b, df["Timestamp"], df[strain_col], label, colour, yaxis="y")
    add_gapped_line(fig_b, temp_plot["Timestamp"], temp_plot[TEMP_COL], "Exhibition Temp", "#000000",
                    width=1.3, yaxis="y2", line_dash="dot", dash_style="dot")
    apply_month_ticks(fig_b, t_min, t_max)
    save_html(fig_b, os.path.join(step1_dir, "strain_with_exhibition_temp_15min.html"))

    fig_c = plotly_base("All Sensors - Strain (15-minute aligned) with Triggered Burst Events",
                        "Bold markers with thin black outline = burst event, per sensor colour. Dashed = gap in the record.")
    fig_c.update_yaxes(title_text="Strain (\u00b5S)")
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        df = updated_sensor_series[sensor_key]
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        burst_col = f"{sensor_key}_Burst(\u00b5S)"
        if df.empty:
            continue
        label = sensor_key.replace("_", " ")
        add_gapped_line(fig_c, df["Timestamp"], df[strain_col], f"{label} (strain)", colour)
        bdf = raw_burst_data[sensor_key]
        if not bdf.empty:
            sensor_cutoff = pd.Timestamp(SETTLING_CUTOFF[folder])
            bdf = bdf[bdf["Timestamp"] >= sensor_cutoff]
        if not bdf.empty:
            events = summarize_burst_events(bdf, "Timestamp", burst_col)
            fig_c.add_trace(go.Scatter(x=events["Timestamp"], y=events[burst_col], mode="markers",
                                       name=f"{label} (burst)", marker=dict(color=colour, size=8,
                                       line=dict(color="black", width=1)),
                                       hovertemplate=f"{label} burst event<br>%{{x|%d %b %Y %H:%M:%S}}<br>peak %{{y:.3f}} \u00b5S<extra></extra>"))
    apply_month_ticks(fig_c, t_min, t_max)
    save_html(fig_c, os.path.join(step1_dir, "strain_with_bursts_15min.html"))

    # ── Steps 2-7 (always full recompute on the current complete dataset) ─
    # The persisted seed/new_seed correctly retains temperature's FULL
    # history (see the outer-merge fix above) so nothing is ever silently
    # dropped from what's saved to disk. But Steps 2-7 must only see the
    # analysis period — the curing period before each sensor's settling
    # cutoff is deliberately excluded project-wide, and Step 3's T0 in
    # particular is a plain mean over every non-null temperature reading
    # regardless of per-sensor strain availability, so it would silently
    # drift to include the pre-settling period if given the untrimmed
    # seed. Filtering to the earliest settling cutoff here (matching the
    # same global_cutoff already used for Step 1's own plots) keeps that
    # out without affecting what's persisted to the seed file itself.
    master_for_analysis = new_seed[new_seed["Timestamp"] >= global_cutoff].reset_index(drop=True)

    step2_dir = get_step_dir(round_dir, "Step 2 Temperature Regression")
    alpha_by_sensor = run_step2(master_for_analysis, step2_dir)

    step3_dir = get_step_dir(round_dir, "Step 3 Compensation")
    combined_compensated, compensated_data = run_step3(master_for_analysis, alpha_by_sensor, step3_dir)

    step4_dir = get_step_dir(round_dir, "Step 4 STL Decomposition")
    all_decomposed, combined_decomposed = run_step4(compensated_data, step4_dir)

    step4b_dir = get_step_dir(round_dir, "Step 4b Daily Noise vs Temperature")
    run_step4b(all_decomposed, master_for_analysis, step4b_dir)

    step4c_dir = get_step_dir(round_dir, "Step 4c Diurnal Profile by Season")
    run_step4c(all_decomposed, step4c_dir)

    step5_dir = get_step_dir(round_dir, "Step 5 Trend Analysis")
    run_step5(all_decomposed, raw_burst_data, step5_dir)

    if combined_compensated is not None:
        step6a_dir = get_step_dir(round_dir, "Step 6a Correlation Analysis (Compensated Strain)")
        run_correlation(combined_compensated,
                        lambda sk: f"{SENSOR_MAP[sk][0]}_Compensated(\u00b5S)"
                        if f"{SENSOR_MAP[sk][0]}_Compensated(\u00b5S)" in combined_compensated.columns else None,
                        step6a_dir, "Compensated Strain")

    if combined_decomposed is not None:
        step6b_dir = get_step_dir(round_dir, "Step 6b Correlation Analysis (Trend)")
        run_correlation(combined_decomposed,
                        lambda sk: f"{SENSOR_MAP[sk][0]}_Trend(\u00b5S)"
                        if f"{SENSOR_MAP[sk][0]}_Trend(\u00b5S)" in combined_decomposed.columns else None,
                        step6b_dir, "Trend")

    step7_dir = get_step_dir(round_dir, "Step 7 Trend Model Fitting")
    run_step7(compensated_data, step7_dir)

    # ── Round summary ────────────────────────────────────────────────────
    run_finished = datetime.now()
    lines = [
        "=" * 70, f"  CANBERRA PR9 SHM - {round_name.upper()} SUMMARY", "=" * 70,
        f"  Run started : {run_started.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Run finished: {run_finished.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Total seed rows: {len(new_seed):,} (+{total_rows_added:,} this round)", "",
        "  ROWS ADDED PER SENSOR (this round):",
    ]
    for sensor_key, n_added in rows_added_by_sensor.items():
        lines.append(f"    {sensor_key.replace('_',' '):<24} +{n_added:,}")
    if continuity_warnings:
        lines += ["", "  CONTINUITY WARNINGS:"]
        for w in continuity_warnings:
            lines.append(f"    !! {w}")
    else:
        lines += ["", "  CONTINUITY: no unexpected gaps or overlaps detected."]
    lines += ["", "=" * 70]
    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    with open(safe_path(os.path.join(round_dir, "round_summary.txt")), "w", encoding="utf-8") as f:
        f.write(summary_text)

    section(f"{round_name.upper()} COMPLETE")
    print(f"  All outputs saved to:\n  {round_dir}")


if __name__ == "__main__":
    main()
