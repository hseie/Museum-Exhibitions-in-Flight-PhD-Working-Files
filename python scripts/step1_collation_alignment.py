"""
step1_collation_alignment.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 1
Collation, 15-minute alignment, gap-fill, settling trim, and raw signal plots.

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors (NP, NS, PB, PT, SB, ST):
  1. Collates all raw Excel exports in that sensor's folder into ONE raw
     strain file (high-rate sheet) and ONE raw burst file (triggered-burst
     sheet), sorted and de-duplicated. The Internal Temperature sheet
     (if present) is ignored. The UTC Timestamp column is dropped — only
     the local wall-clock "Time" column is kept.
  2. Aligns the raw strain data to a 15-minute grid (nearest-neighbour
     snapping, 8-minute max tolerance). Burst data is NOT aligned — it is
     kept at native (event) resolution.
  3. Gap-fills the aligned strain data (forward/backward fill meeting in
     the middle, only for gaps of <= 40 consecutive missing steps AND
     <= 5 microstrain edge difference).
  4. Applies the 30-day settling trim, per sensor, from its own install
     date:
         NP, NS, PB, PT, SB : keep from 2025-06-14 onward
         ST                  : keep from 2025-09-27 onward
Also loads and 15-min-aligns the exhibition temperature record.

OUTPUTS (all saved to Output\\Round 1\\Step 1 Collation Alignment and Raw
Plots\\, per the project's fixed convention: numbered round, numbered/named
step subfolder):

  Per sensor:
    <Sensor>_raw_strain.csv                   - collated raw strain (native res.)
    <Sensor>_raw_burst.csv                    - collated raw burst (native res.)
    <Sensor>_aligned_15min.csv                - aligned, before gap-fill/trim
    <Sensor>_aligned_15min_filled_trimmed.csv - aligned + gap-filled + trimmed
                                                 (this is what later steps use)

  Combined:
    Exhibition_Temp_aligned_15min.csv
    All_Sensors_Aligned_Filled_Trimmed.csv    - all 6 sensors + temp, merged

  Plots (HTML, Plotly, all interactive with range slider + toggleable legend).
  These use the 15-minute ALIGNED strain (post gap-fill/settling-trim), not
  the native-resolution raw collation — the full monitoring period at native
  ~15-second resolution would produce impractically large HTML files:
    strain_all_sensors_15min.html
    strain_with_exhibition_temp_15min.html    - strain left axis, temp right axis
    strain_with_bursts_15min.html              - burst events as markers
                                                  (bursts themselves stay at
                                                  native/event resolution —
                                                  they're sparse markers, not
                                                  a continuous high-rate line)

GLOBAL PLOT CONVENTIONS (apply to every time-series plot in this pipeline):
  - Date axis ticks at the 1st and 15th of every month spanned by the data
  - Range slider visible at the bottom of the x-axis
  - Legend entries individually toggleable (Plotly default — click to hide/show)
  - Fixed sensor colours, used consistently

USAGE
=====
    Edit BASE_DIR below if the project folder ever moves, then run:
        python step1_collation_alignment.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds — save
edits as step1_collation_alignment_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# Prefer the 'calamine' engine for reading .xlsx — it is roughly 5x faster
# than openpyxl on large sheets (tested: ~600k-row sheet: 28s vs 5s), which
# matters here since each sensor has 11 files spanning 14+ months. Falls
# back to openpyxl automatically if python-calamine isn't installed.
# Install with:  pip install python-calamine
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

SENSORS_SUBDIR    = "Sensors"                  # contains NP\, NS\, PB\, PT\, SB\, ST\
EXHIBITION_SUBDIR = "Exhibition Environment"    # contains EH5_Temp.csv (and EH5_RH.csv)

OUTPUT_BASE  = "Output"
ROUND_NAME   = "Round 1"
STEP_FOLDER  = "Step 1 Collation Alignment and Raw Plots"

FONT = "Arial"

# Sensor subfolder -> (internal key used in column names / filenames, plot colour)
# Colours match the project's fixed convention — do not change.
SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

# 30-day settling trim cutoff, per sensor (fixed dates, confirmed with the
# project lead — install date + 30 days, except where curing-period data
# was already excluded before the cutoff was set).
SETTLING_CUTOFF = {
    "NP": "2025-06-14",
    "NS": "2025-06-14",
    "PB": "2025-06-14",
    "PT": "2025-06-14",
    "SB": "2025-06-14",
    "ST": "2025-09-27",
}

# Fixed pipeline parameters — reuse, do not re-derive.
SNAP_TOLERANCE_MIN = 8      # max minutes between raw reading and 15-min grid point
GAP_FILL_MAX_STEPS = 40     # max consecutive missing 15-min steps to fill (=10 hrs)
GAP_FILL_MAX_EDGE  = 5.0    # max microstrain difference across a gap before leaving as NaN

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


def get_output_dir():
    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    os.makedirs(safe_path(out_dir), exist_ok=True)
    return out_dir


def safe_path(path):
    """
    Windows has a classic 260-character MAX_PATH limit, which a deep
    OneDrive folder structure combined with this pipeline's longer output
    filenames can exceed (confirmed: some output paths here run to ~260+
    characters). Prepending the '\\\\?\\' extended-length prefix bypasses
    that limit at the Win32 API level for absolute paths — this needs no
    admin rights and no system/registry changes, unlike the newer
    "Enable Win32 long paths" policy. No effect on non-Windows systems.
    """
    if os.name == "nt":
        abs_path = os.path.abspath(path)
        if not abs_path.startswith("\\\\?\\"):
            abs_path = "\\\\?\\" + abs_path
        return abs_path
    return path


def find_sheet(sheet_names, keyword):
    """Return the first sheet name containing keyword (case-insensitive), or None."""
    for name in sheet_names:
        if keyword.lower() in name.lower():
            return name
    return None


def classify_sheet(xls, sheet_name):
    """
    Classify a sheet as 'strain', 'burst', or 'internal' by reading only
    its header row and checking the actual column header text — NOT the
    sheet name, and NOT sheet position/order.

    Neither of those is reliable: Excel truncates sheet names to 31
    characters (cutting keywords like 'high' or 'trigg' short for longer
    sensor names), AND sheet order is not consistent across all of a
    sensor's source files (confirmed: one file had the high-rate strain
    sheet in position 3, not position 1). Column header text is a plain
    cell value, so it is never truncated and doesn't depend on order.
    """
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


# ── Step 1a: collate raw sensor files ────────────────────────────────────────

def load_sensor_files(sensor_folder_path, sensor_key):
    """
    Reads every Excel file in a sensor's folder, pulls out the high-rate
    strain sheet and the triggered-burst sheet from each (Internal
    Temperature sheet, if present, is ignored), and returns two collated,
    sorted, de-duplicated DataFrames: (strain_df, burst_df).
    Only the Time and value columns are kept — UTC Timestamp is dropped.
    """
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

        # Classify every sheet in this file by its actual column header
        # text — NOT by name or position, since neither is reliable across
        # this sensor's 11 source files (see classify_sheet docstring).
        sheet_high = sheet_burst = None
        for name in xls.sheet_names:
            kind = classify_sheet(xls, name)
            if kind == "strain" and sheet_high is None:
                sheet_high = name
            elif kind == "burst" and sheet_burst is None:
                sheet_burst = name
            # kind == "internal" (or unrecognised) deliberately skipped —
            # internal equipment temperature is out of scope this round.

        if sheet_high is None:
            warn(f"{os.path.basename(f)}: no high-rate strain sheet identified — file skipped for strain")
        if sheet_burst is None:
            # Normal — many files have no burst sheet at all because no
            # triggered burst occurred in that period. Not a warning.
            pass

        if sheet_high is not None:
            df = pd.read_excel(xls, sheet_name=sheet_high)
            df = df.iloc[:, :2].copy()          # Time, Value (drops UTC Timestamp)
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
       f"{len(burst_df):,} raw burst rows collated from {len(files)} file(s)")
    return strain_df, burst_df


# ── Step 1b: align to 15-minute grid ────────────────────────────────────────

def align_to_15min(df, ts_col, val_col, tolerance_min=SNAP_TOLERANCE_MIN):
    """Nearest-neighbour snap onto a regular 15-minute grid."""
    d = df[[ts_col, val_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame(columns=["Timestamp", val_col])
    d = d.sort_values(ts_col)
    d[ts_col] = d[ts_col].dt.round("s")
    t_min = d[ts_col].min().floor("15min")
    t_max = d[ts_col].max().ceil("15min")
    grid = pd.DataFrame({"Timestamp": pd.date_range(t_min, t_max, freq="15min")})
    merged = pd.merge_asof(
        grid, d.rename(columns={ts_col: "Timestamp", val_col: "value"}),
        on="Timestamp",
        tolerance=pd.Timedelta(minutes=tolerance_min),
        direction="nearest",
    )
    merged = merged.rename(columns={"value": val_col})
    return merged


# ── Step 1c: gap-fill ────────────────────────────────────────────────────────

def gap_fill(series, max_steps=GAP_FILL_MAX_STEPS, max_edge=GAP_FILL_MAX_EDGE):
    """Forward/backward fill meeting in the middle, bounded by duration and edge."""
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


# ── Step 1d: settling trim ──────────────────────────────────────────────────

def apply_settling_trim(df, ts_col, cutoff_str):
    cutoff = pd.Timestamp(cutoff_str)
    return df[df[ts_col] >= cutoff].reset_index(drop=True)


# ── Burst event summarization (for plotting only) ───────────────────────────

def summarize_burst_events(df, ts_col, val_col, gap_seconds=2.0):
    """
    Collapse dense burst readings into one marker per distinct burst EVENT,
    for plotting only (the full-resolution burst CSV is unaffected).

    A single triggered burst spans 9-15 seconds at 10-100 samples/sec, so
    one 'event' can contain hundreds of individual readings — plotting
    every one of them produced an 8-9 MB, slow-loading HTML page from a
    single sample file, and would scale to tens of MB across the full
    dataset. Readings within `gap_seconds` of each other are treated as
    the same event; each event is represented by its single most extreme
    (max absolute value) reading, which is the physically meaningful
    point of a burst.
    """
    if df.empty:
        return df
    d = df[[ts_col, val_col]].dropna().sort_values(ts_col).reset_index(drop=True)
    gaps = d[ts_col].diff().dt.total_seconds()
    event_id = (gaps > gap_seconds).cumsum()
    idx = d.groupby(event_id)[val_col].apply(lambda s: s.abs().idxmax())
    return d.loc[idx].reset_index(drop=True)


# ── Exhibition temperature ──────────────────────────────────────────────────

def load_exhibition_temp(exhibition_dir):
    candidates = [
        f for f in glob.glob(os.path.join(exhibition_dir, "*.csv"))
        if "temp" in os.path.basename(f).lower()
        and "rh" not in os.path.basename(f).lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No exhibition temperature CSV found in {exhibition_dir}"
        )
    f = candidates[0]
    df = pd.read_csv(f, encoding="utf-8-sig")
    df.columns = ["Timestamp", "Exhibition_Temp(\u00b0C)"]

    # Strip trailing timezone abbreviation (e.g. "BST") then parse.
    ts_clean = (
        df["Timestamp"].astype(str).str.strip()
        .str.replace(r"\s+[A-Za-z]{2,4}$", "", regex=True)
    )
    parsed = pd.to_datetime(ts_clean, format="%d-%b-%y %I:%M:%S %p", errors="coerce")
    if parsed.isna().mean() > 0.01:
        parsed = pd.to_datetime(ts_clean, errors="coerce", dayfirst=True)
    df["Timestamp"] = parsed
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    ok(f"Exhibition temperature loaded: {len(df):,} rows from {os.path.basename(f)}")
    return df


# ── Plotting helpers ─────────────────────────────────────────────────────────

def month_ticks(t_min, t_max):
    """List of the 1st and 15th of every month spanned by [t_min, t_max]."""
    ticks = []
    cur = pd.Timestamp(t_min.year, t_min.month, 1)
    end = pd.Timestamp(t_max.year, t_max.month, 1)
    while cur <= end:
        if cur >= pd.Timestamp(t_min.date()) - pd.Timedelta(days=31):
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
        tickmode="array",
        tickvals=ticks,
        ticktext=[t.strftime("%d %b %Y") for t in ticks],
        tickangle=-45,
    )


def add_gapped_line(fig, x, y, name, colour, width=0.8, yaxis="y",
                     legendgroup=None, hovertemplate=None, dash_style="dash",
                     line_dash=None):
    """
    Add a time-series line with the project's fixed gap convention: real
    data as a solid line (gaps left blank, connectgaps=False), plus a
    second, faded, dashed 'halftone' trace bridging the same gaps
    (connectgaps=True) so a missing stretch reads visually as a gap
    rather than a blank or a misleadingly straight interpolation.
    `line_dash` optionally styles the main (real-data) trace itself
    (e.g. a dotted temperature line to distinguish it from solid strain
    lines) — independent of `dash_style`, which is only for the faded
    gap-bridging trace.
    """
    group = legendgroup or name
    main_line = dict(color=colour, width=width)
    if line_dash:
        main_line["dash"] = line_dash
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=name,
        line=main_line,
        connectgaps=False, legendgroup=group, yaxis=yaxis,
        hovertemplate=hovertemplate,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=f"{name} (gap)",
        line=dict(color=colour, width=width, dash=dash_style),
        connectgaps=True, opacity=0.3, legendgroup=group, yaxis=yaxis,
        showlegend=False, hoverinfo="skip",
    ))


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


def save_html(fig, path):
    fig.write_html(safe_path(path), include_plotlyjs="cdn")
    ok(f"Saved: {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = get_output_dir()
    print(f"Output folder: {out_dir}")

    sensor_data = {}   # sensor_key -> dict(colour, raw_strain, raw_burst, trimmed)

    section("STEP 1 - Collation, alignment, gap-fill, settling trim (per sensor)")
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        sensor_path = os.path.join(BASE_DIR, SENSORS_SUBDIR, folder)
        if not os.path.isdir(sensor_path):
            warn(f"Sensor folder not found, skipped: {sensor_path}")
            continue

        raw_strain, raw_burst = load_sensor_files(sensor_path, sensor_key)

        raw_strain.to_csv(safe_path(os.path.join(out_dir, f"{sensor_key}_raw_strain.csv")), index=False, encoding="utf-8-sig")
        raw_burst.to_csv(safe_path(os.path.join(out_dir, f"{sensor_key}_raw_burst.csv")), index=False, encoding="utf-8-sig")

        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        aligned = align_to_15min(raw_strain, "Timestamp", strain_col)
        aligned.to_csv(safe_path(os.path.join(out_dir, f"{sensor_key}_aligned_15min.csv")), index=False, encoding="utf-8-sig")

        aligned[strain_col] = gap_fill(aligned[strain_col])

        cutoff = SETTLING_CUTOFF[folder]
        trimmed = apply_settling_trim(aligned, "Timestamp", cutoff)
        trimmed.to_csv(
            safe_path(os.path.join(out_dir, f"{sensor_key}_aligned_15min_filled_trimmed.csv")),
            index=False, encoding="utf-8-sig",
        )
        ok(f"{sensor_key}: aligned {len(aligned):,} rows -> "
           f"trimmed to {len(trimmed):,} rows (cutoff {cutoff})")

        sensor_data[sensor_key] = dict(
            colour=colour, raw_strain=raw_strain, raw_burst=raw_burst, trimmed=trimmed,
        )

    if not sensor_data:
        raise SystemExit("No sensor data loaded — check BASE_DIR / SENSORS_SUBDIR.")

    section("Exhibition temperature")
    temp_df = load_exhibition_temp(os.path.join(BASE_DIR, EXHIBITION_SUBDIR))
    temp_aligned = align_to_15min(temp_df, "Timestamp", "Exhibition_Temp(\u00b0C)")
    temp_aligned.to_csv(
        safe_path(os.path.join(out_dir, "Exhibition_Temp_aligned_15min.csv")), index=False,
        encoding="utf-8-sig",
    )

    section("Combined master file (all sensors + temperature)")
    master = None
    for sensor_key, d in sensor_data.items():
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        df = d["trimmed"][["Timestamp", strain_col]]
        master = df if master is None else pd.merge(master, df, on="Timestamp", how="outer")
    master = pd.merge(master, temp_aligned, on="Timestamp", how="left")
    master = master.sort_values("Timestamp").reset_index(drop=True)
    master.to_csv(
        safe_path(os.path.join(out_dir, "All_Sensors_Aligned_Filled_Trimmed.csv")), index=False,
        encoding="utf-8-sig",
    )
    ok(f"Combined master file saved: {len(master):,} rows")

    # ── Overall time span (for tick generation) across the aligned data ──
    t_min = min(d["trimmed"]["Timestamp"].min() for d in sensor_data.values() if not d["trimmed"].empty)
    t_max = max(d["trimmed"]["Timestamp"].max() for d in sensor_data.values() if not d["trimmed"].empty)

    # Global settling cutoff for shared (non-per-sensor) series, e.g.
    # exhibition temperature — the earliest of all sensors' individual
    # settling cutoffs (2025-06-14 for every sensor except Starboard Top).
    # This convention (excluding pre-settling data from plots) applies to
    # every step in this pipeline from here on, not just Step 1.
    global_cutoff = pd.Timestamp(min(SETTLING_CUTOFF.values()))
    temp_plot = temp_aligned[temp_aligned["Timestamp"] >= global_cutoff]

    section("Plots")

    # === Plot A: strain signal, all sensors, 15-min aligned ===
    fig_a = plotly_base(
        "All Sensors - Strain (15-minute aligned, gap-filled, settling-trimmed)",
        "Pre-compensation strain, per sensor. Dashed = gap in the record.",
    )
    fig_a.update_yaxes(title_text="Strain (\u00b5S)")
    for sensor_key, d in sensor_data.items():
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        df = d["trimmed"]
        label = sensor_key.replace("_", " ")
        add_gapped_line(
            fig_a, df["Timestamp"], df[strain_col], label, d["colour"],
            hovertemplate=f"{label}<br>%{{x|%d %b %Y %H:%M}}<br>%{{y:.3f}} \u00b5S<extra></extra>",
        )
    apply_month_ticks(fig_a, t_min, t_max)
    save_html(fig_a, os.path.join(out_dir, "strain_all_sensors_15min.html"))

    # === Plot B: strain + exhibition temperature (dual axis), 15-min aligned ===
    fig_b = plotly_base(
        "All Sensors - Strain with Exhibition Temperature (15-minute aligned)",
        "Strain on left axis (per sensor), exhibition temperature on right axis. Dashed = gap in the record.",
    )
    fig_b.update_layout(
        yaxis=dict(title="Strain (\u00b5S)", showgrid=True, gridcolor="#eeeeee"),
        yaxis2=dict(title="Exhibition Temp (\u00b0C)", overlaying="y",
                     side="right", showgrid=False),
    )
    for sensor_key, d in sensor_data.items():
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        df = d["trimmed"]
        label = sensor_key.replace("_", " ")
        add_gapped_line(fig_b, df["Timestamp"], df[strain_col], label, d["colour"], yaxis="y")
    add_gapped_line(
        fig_b, temp_plot["Timestamp"], temp_plot["Exhibition_Temp(\u00b0C)"],
        "Exhibition Temp", "#000000", width=1.3, yaxis="y2",
        line_dash="dot", dash_style="dot",
    )
    apply_month_ticks(fig_b, t_min, t_max)
    save_html(fig_b, os.path.join(out_dir, "strain_with_exhibition_temp_15min.html"))

    # === Plot C: strain (15-min aligned) + burst events overlay ===
    # Burst events are kept at native (event) resolution and summarized to
    # one marker per event (see summarize_burst_events) — they are sparse
    # markers, not a continuous line, so gap-dashing doesn't apply to them.
    # Each sensor's own bursts are filtered to that sensor's own settling
    # cutoff, matching how its strain data is already trimmed.
    fig_c = plotly_base(
        "All Sensors - Strain (15-minute aligned) with Triggered Burst Events",
        "Bold markers with thin black outline = burst event, per sensor colour. Dashed = gap in the record.",
    )
    fig_c.update_yaxes(title_text="Strain (\u00b5S)")
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        if sensor_key not in sensor_data:
            continue
        d = sensor_data[sensor_key]
        strain_col = f"{sensor_key}_Strain(\u00b5S)"
        burst_col  = f"{sensor_key}_Burst(\u00b5S)"
        df = d["trimmed"]
        label = sensor_key.replace("_", " ")
        add_gapped_line(fig_c, df["Timestamp"], df[strain_col], f"{label} (strain)", d["colour"])

        bdf = d["raw_burst"]
        if not bdf.empty:
            sensor_cutoff = pd.Timestamp(SETTLING_CUTOFF[folder])
            bdf = bdf[bdf["Timestamp"] >= sensor_cutoff]
        if not bdf.empty:
            events = summarize_burst_events(bdf, "Timestamp", burst_col)
            fig_c.add_trace(go.Scatter(
                x=events["Timestamp"], y=events[burst_col],
                mode="markers", name=f"{label} (burst)",
                marker=dict(color=d["colour"], size=8,
                             line=dict(color="black", width=1)),
                hovertemplate=f"{label} burst event<br>%{{x|%d %b %Y %H:%M:%S}}<br>peak %{{y:.3f}} \u00b5S<extra></extra>",
            ))
    apply_month_ticks(fig_c, t_min, t_max)
    save_html(fig_c, os.path.join(out_dir, "strain_with_bursts_15min.html"))


    section("STEP 1 COMPLETE")
    print(f"  All outputs saved to:\n  {out_dir}")


if __name__ == "__main__":
    main()
