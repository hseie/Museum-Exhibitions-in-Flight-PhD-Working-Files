"""
step11b_year_over_year_trend_events_bursts.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 11b (standalone, NOT part of run_full_pipeline.py)
Overlays 2025 and 2026's trend on the same calendar window (mid-June to
mid-August), with that year's burst events and recorded site events plotted
on top - so you can see whether the trend's response to real events (or
lack of trend movement generally) differed between the two years.

WHERE THIS SITS
=================
Reads Step 4's per-sensor trend, raw burst data straight from the original
per-sensor Excel files (same source Step 5 reads, same sheet-classification
logic), and the recorded-events workbook. Standalone for now - not wired
into run_full_pipeline.py.

EVENTS SHOWN ONLY IN THE YEAR THEY HAPPENED
==============================================
A recorded event or burst only occurred once, in one specific year - it has
no real counterpart in the other year. Both years' trend lines are drawn
overlaid, but events/bursts appear only at their own real calendar position
(re-expressed as elapsed days since the window start), so 2025 may show
events with nothing at the same elapsed-day position on 2026, and vice
versa. This is the accurate representation, not a limitation to fix.

OVERLAP WINDOW
================
14 June to 13 August (inclusive), by month-day, ignoring year - matches
Step 11a.

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors, produces ONE file: 2025's trend (light
colour) and 2026's trend (full colour) on the same elapsed-day x-axis, with
that year's burst events marked on its own trend line, and recorded events
shown as shaded spans / dashed lines, coloured by category.

OUTPUTS (saved to Output\\Round 2\\Step 11 Year over Year Comparison\\):
  <Sensor>_yoy_trend_events_bursts.html   - one file per sensor

USAGE
=====
    Run AFTER step4_stl_decomposition.py.
        python step11b_year_over_year_trend_events_bursts.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds - save
edits as step11b_year_over_year_trend_events_bursts_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import glob
import warnings

import pandas as pd
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

OUTPUT_BASE   = "Output"
ROUND_NAME    = "Round 2"
STEP4_FOLDER  = "Step 4 STL Decomposition"
STEP_FOLDER   = "Step 11 Year over Year Comparison"
SENSORS_SUBDIR = "Sensors"

EVENTS_FILE = os.path.join(BASE_DIR, "Exhibition Environment", "01 Canberra Recorded Events.xlsx")

FONT = "Arial"

SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

EVENT_CATEGORY_COLOURS = {
    "weather": "#ED7D31", "around_aircraft": "#4472C4", "interference": "#FFC000",
}
THEME_INDEX_TO_CATEGORY = {5: "weather", 8: "around_aircraft", 7: "interference"}
CATEGORY_LABELS = {"weather": "Weather event", "around_aircraft": "Event around the Canberra",
                   "interference": "Stepping on / interference"}

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
    ref_start = pd.Timestamp(2001, *WINDOW_START_MONTHDAY)
    ref_end = pd.Timestamp(2001, *WINDOW_END_MONTHDAY)
    total_days = (ref_end - ref_start).days
    tickvals = list(range(0, total_days + 1, 7))
    if tickvals[-1] != total_days:
        tickvals.append(total_days)
    ticktext = [(ref_start + pd.Timedelta(days=d)).strftime("%d %b") for d in tickvals]
    return tickvals, ticktext


def elapsed_days(ts_series, year):
    start, _ = window_bounds(year)
    return (ts_series - start).dt.total_seconds() / 86400


def classify_sheet(xls, sheet_name):
    try:
        header_df = pd.read_excel(xls, sheet_name=sheet_name, nrows=0)
    except Exception:
        return None
    cols = " ".join(str(c) for c in header_df.columns).lower()
    if "triggered" in cols or "trigger" in cols or "burst" in cols:
        return "burst"
    return None


def load_burst_data(sensor_folder_path, sensor_key):
    files = sorted(
        glob.glob(os.path.join(sensor_folder_path, "*.xlsx")) +
        glob.glob(os.path.join(sensor_folder_path, "*.xls"))
    )
    if not files:
        warn(f"No Excel files found in {sensor_folder_path}")
        return pd.DataFrame(columns=["Timestamp", f"{sensor_key}_Burst(\u00b5S)"])

    burst_col = f"{sensor_key}_Burst(\u00b5S)"
    frames = []
    for f in files:
        try:
            xls = pd.ExcelFile(f, engine=EXCEL_ENGINE)
        except Exception as e:
            warn(f"Could not open {os.path.basename(f)}: {e}")
            continue
        sheet_burst = None
        for name in xls.sheet_names:
            if classify_sheet(xls, name) == "burst":
                sheet_burst = name
                break
        if sheet_burst is not None:
            df = pd.read_excel(xls, sheet_name=sheet_burst)
            df = df.iloc[:, :2].copy()
            df.columns = ["Timestamp", burst_col]
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Timestamp", burst_col])
    out = pd.concat(frames, ignore_index=True)
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    out = out.dropna(subset=["Timestamp"]).drop_duplicates(subset="Timestamp")
    out = out.sort_values("Timestamp").reset_index(drop=True)
    return out


def summarize_burst_events(df, ts_col, val_col, gap_seconds=2.0):
    if df.empty:
        return df
    d = df[[ts_col, val_col]].dropna().sort_values(ts_col).reset_index(drop=True)
    gaps = d[ts_col].diff().dt.total_seconds()
    event_id = (gaps > gap_seconds).cumsum()
    idx = d.groupby(event_id)[val_col].apply(lambda s: s.abs().idxmax())
    return d.loc[idx].reset_index(drop=True)


def load_recorded_events(path):
    if not os.path.isfile(safe_path(path)):
        warn(f"Recorded-events file not found: {path}")
        return []
    try:
        wb = openpyxl.load_workbook(safe_path(path), data_only=True)
    except Exception as e:
        warn(f"Could not read recorded-events file ({e})")
        return []
    ws = wb.worksheets[0]
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
        events.append({"date": date, "category": category, "block_id": block_id})
    ok(f"Recorded events loaded: {len(events)} dated row(s) categorised")
    return events


def main():
    section("Step 11b: Year-over-Year Trend with Bursts and Recorded Events")

    step4_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP4_FOLDER)
    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    try:
        os.makedirs(safe_path(out_dir), exist_ok=True)
    except PermissionError:
        warn(f"Could not create output folder (permission denied): {out_dir}")
        return

    all_events = load_recorded_events(EVENTS_FILE)
    groups = {}
    for ev in all_events:
        groups.setdefault((ev["block_id"], ev["category"]), []).append(ev)
    span_events = {k: v for k, v in groups.items() if len(v) >= 2}
    consumed_ids = set()
    for members in span_events.values():
        consumed_ids.update(id(m) for m in members)

    for code, (sensor_name, colour) in SENSOR_MAP.items():
        section(sensor_name)
        decomp_path = safe_path(os.path.join(step4_dir, f"{sensor_name}_decomposed.csv"))
        if not os.path.exists(decomp_path):
            warn(f"Decomposed file not found: {decomp_path} - skipping.")
            continue
        decomp = pd.read_csv(decomp_path, parse_dates=["Timestamp"], encoding="utf-8-sig")
        trend_col = f"{sensor_name}_Trend(\u00b5S)"
        if not {"Timestamp", trend_col}.issubset(decomp.columns):
            warn(f"{sensor_name}: expected column '{trend_col}' not found - skipping.")
            continue
        trend_df = decomp[["Timestamp", trend_col]].dropna()

        sensor_folder = safe_path(os.path.join(BASE_DIR, SENSORS_SUBDIR, code))
        burst_df = load_burst_data(sensor_folder, sensor_name)
        burst_col = f"{sensor_name}_Burst(\u00b5S)"

        light_colour = lighten_hex(colour)
        year_styles = {YEARS[0]: (light_colour, "2025"), YEARS[1]: (colour, "2026")}

        # ── Fixed event-band boundaries, computed once from the trend's own
        # ── in-window range so the bands sit just outside the data, not
        # ── overlapping it. 2025's events live in a thin strip BELOW the
        # ── data; 2026's events live in a thin strip ABOVE it - so the year
        # ── is conveyed by vertical position, not by line style.
        window_vals = []
        for year in YEARS:
            start, end = window_bounds(year)
            window_vals.append(trend_df.loc[(trend_df["Timestamp"] >= start) &
                                             (trend_df["Timestamp"] <= end), trend_col])
        window_vals = pd.concat(window_vals) if window_vals else pd.Series(dtype=float)
        if window_vals.empty:
            warn(f"{sensor_name}: no trend data found for either year - skipping file.")
            continue
        y_min, y_max = window_vals.min(), window_vals.max()
        data_range = max(y_max - y_min, 1e-6)
        band_gap = 0.06 * data_range
        band_thick = 0.10 * data_range
        band_2025_top = y_min - band_gap
        band_2025_bottom = band_2025_top - band_thick
        band_2026_bottom = y_max + band_gap
        band_2026_top = band_2026_bottom + band_thick
        band_mid = {YEARS[0]: (band_2025_top + band_2025_bottom) / 2,
                    YEARS[1]: (band_2026_top + band_2026_bottom) / 2}
        band_range = {YEARS[0]: (band_2025_bottom, band_2025_top),
                      YEARS[1]: (band_2026_bottom, band_2026_top)}

        fig = go.Figure()
        any_data = False
        seen_categories = {}

        for year in YEARS:
            start, end = window_bounds(year)
            yt = trend_df[(trend_df["Timestamp"] >= start) & (trend_df["Timestamp"] <= end)].copy()
            if yt.empty:
                warn(f"{sensor_name} / {year}: no trend data in window - skipped.")
                continue
            any_data = True
            yt["Elapsed_Days"] = elapsed_days(yt["Timestamp"], year)
            line_colour, label = year_styles[year]
            fig.add_trace(go.Scatter(
                x=yt["Elapsed_Days"], y=yt[trend_col], mode="lines",
                line=dict(color=line_colour, width=1.3), name=f"Trend {label}",
                legendgroup=label,
                customdata=yt["Timestamp"],
                hovertemplate=f"Trend {label}<br>%{{customdata|%d %b %Y %H:%M}}"
                              f"<br>%{{y:.3f}} \u00b5S<extra></extra>",
            ))

            # ── Burst events for this year, within the window ──────────
            if not burst_df.empty and burst_col in burst_df.columns:
                yb = burst_df[(burst_df["Timestamp"] >= start) & (burst_df["Timestamp"] <= end)]
                if not yb.empty:
                    events = summarize_burst_events(yb, "Timestamp", burst_col)
                    if not events.empty:
                        merged = pd.merge_asof(
                            events.sort_values("Timestamp"), yt.sort_values("Timestamp"),
                            on="Timestamp", direction="nearest", tolerance=pd.Timedelta(hours=6),
                        ).dropna(subset=[trend_col])
                        if not merged.empty:
                            merged["Elapsed_Days"] = elapsed_days(merged["Timestamp"], year)
                            fig.add_trace(go.Scatter(
                                x=merged["Elapsed_Days"], y=merged[trend_col], mode="markers",
                                marker=dict(color=line_colour, size=9, line=dict(color="black", width=1)),
                                name=f"Burst {label}", legendgroup=label, showlegend=False,
                                hovertemplate=f"Burst event {label}<br>%{{customdata|%d %b %Y %H:%M}}<extra></extra>",
                                customdata=merged["Timestamp"],
                            ))

            # ── Recorded events for this year, within the window ───────
            # Year is conveyed by vertical position: 2025's events sit in a
            # thin band BELOW the trend data, 2026's sit in a thin band
            # ABOVE it - both bands fixed regardless of x-position, so
            # comparing the same week across years is a direct up/down read.
            band_bottom, band_top = band_range[year]
            for (block_id, category), members in span_events.items():
                members_in_year = [m for m in members if start <= m["date"] <= end]
                if len(members_in_year) < 2:
                    continue
                x0 = elapsed_days(pd.Series([min(m["date"] for m in members_in_year)]), year).iloc[0]
                x1 = elapsed_days(pd.Series([max(m["date"] for m in members_in_year) + pd.Timedelta(days=1)]), year).iloc[0]
                fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1,
                             y0=band_bottom, y1=band_top,
                             fillcolor=EVENT_CATEGORY_COLOURS[category], opacity=0.75, line_width=0)
                seen_categories[(category, year)] = True

            single_day_x, single_day_colour, single_day_date, single_day_label = [], [], [], []
            for ev in all_events:
                if id(ev) in consumed_ids:
                    continue
                if not (start <= ev["date"] <= end):
                    continue
                x_pos = elapsed_days(pd.Series([ev["date"]]), year).iloc[0]
                single_day_x.append(x_pos)
                single_day_colour.append(EVENT_CATEGORY_COLOURS[ev["category"]])
                single_day_date.append(ev["date"])
                single_day_label.append(CATEGORY_LABELS[ev["category"]])
                seen_categories[(ev["category"], year)] = True

            if single_day_x:
                fig.add_trace(go.Scatter(
                    x=single_day_x, y=[band_mid[year]] * len(single_day_x), mode="markers",
                    marker=dict(symbol="square", size=10, color=single_day_colour,
                               line=dict(color="black", width=0.5)),
                    showlegend=False,
                    customdata=list(zip(single_day_date, single_day_label)),
                    hovertemplate="%{customdata[1]}<br>%{customdata[0]|%d %b %Y}<extra></extra>",
                ))

        if not any_data:
            warn(f"{sensor_name}: no data found in the overlap window - skipping file.")
            continue

        for (cat, year), _ in seen_categories.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                     marker=dict(symbol="square", size=10, color=EVENT_CATEGORY_COLOURS[cat],
                                                line=dict(color="black", width=0.5)),
                                     name=f"{CATEGORY_LABELS[cat]} ({year})"))

        # Small text labels on the bands themselves so the year-position
        # convention (2025 below, 2026 above) is obvious without checking
        # the legend.
        fig.add_annotation(x=0, y=band_mid[YEARS[0]], text="2025 events", showarrow=False,
                           xanchor="right", xshift=-8, font=dict(size=9, color="#888888"))
        fig.add_annotation(x=0, y=band_mid[YEARS[1]], text="2026 events", showarrow=False,
                           xanchor="right", xshift=-8, font=dict(size=9, color="#888888"))

        fig.update_layout(
            title=f"{sensor_name}: Year-over-Year Trend with Bursts and Recorded Events "
                  f"({pd.Timestamp(2000,*WINDOW_START_MONTHDAY).strftime('%d %b')}"
                  f"\u2013{pd.Timestamp(2000,*WINDOW_END_MONTHDAY).strftime('%d %b')})",
            font=dict(family=FONT, size=11), height=620,
            xaxis_title="Date (both years overlaid on the same calendar position)",
            yaxis_title="Trend (\u00b5S)",
            legend=dict(orientation="v", y=1, x=1.02, xanchor="left", yanchor="top",
                       font=dict(size=10), bgcolor="rgba(255,255,255,0.95)",
                       bordercolor="#cccccc", borderwidth=1),
            margin=dict(l=65, r=190, t=70, b=110), plot_bgcolor="white",
        )
        tickvals, ticktext = elapsed_day_ticks()
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext,
                         showgrid=True, gridcolor="#eeeeee", rangeslider_visible=True)
        fig.update_yaxes(showgrid=True, gridcolor="#eeeeee",
                         range=[band_2025_bottom - band_gap * 0.5, band_2026_top + band_gap * 0.5])

        html_path = safe_path(os.path.join(out_dir, f"{sensor_name}_yoy_trend_events_bursts.html"))
        try:
            fig.write_html(html_path, include_plotlyjs="cdn")
            ok(f"Saved: {os.path.basename(html_path)}")
        except PermissionError:
            warn(f"Could not write plot (file open elsewhere?): {html_path}")

    section("Done")


if __name__ == "__main__":
    main()
