"""
step5_trend_analysis.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 5
Trend analysis: all sensors' STL trends combined on one plot, and a second
version of that same plot with triggered-burst events and recorded
site/weather events overlaid.

WHAT THIS SCRIPT DOES
======================
  1. Loads each sensor's Trend column from Step 4's decomposed output.
  2. Plot 1 ("trend_all_sensors.html"): all six sensors' trends on one
     combined plot — clean comparison, no bursts, no events.
  3. Plot 2 ("trend_with_bursts_and_events.html"): the same combined
     trend plot, with:
       - one marker per burst EVENT per sensor (not every reading —
         reusing the same event-grouping approach as Step 1), placed at
         that sensor's own trend value at the burst's timestamp
       - vertical lines / shaded spans for every dated row in the
         recorded-events spreadsheet, coloured by category (Weather /
         Around the Canberra / Interference) using that spreadsheet's
         own fill colours

INPUT: the recorded-events spreadsheet
=======================================
EVENTS_FILE points to "01 Canberra Recorded Events.xlsx" inside the
Exhibition Environment folder. If that path ever moves, this script still
runs Plot 1 and the burst overlay in Plot 2, but skips event highlighting
with a warning rather than failing outright.

Each row with its own Date is read as one event, categorised by that
row's fill colour (not by which column has text). Rows that only
continue a previous event's description without their own date (a few
exist in the source spreadsheet) are NOT picked up — only dated rows.

OUTPUTS (saved to Output\\<latest Round N>\\Step 5 Trend Analysis\\ —
see ROUND_NAME in the config section if you need to target a specific
round instead of always using the latest one):
  trend_all_sensors.html
  trend_with_bursts_and_events.html

USAGE
=====
    Run AFTER step1_collation_alignment.py and step4_stl_decomposition.py.
        python step5_trend_analysis.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds — save
edits as step5_trend_analysis_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import warnings

import numpy as np
import pandas as pd
import openpyxl
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

# Recorded-events spreadsheet — lives inside the Exhibition Environment
# folder alongside the exhibition temperature/RH data.
EVENTS_FILE = os.path.join(
    r"C:\Users\c2051290\OneDrive - Newcastle University\NCLU\PhD\My PhD"
    r"\01 Onsite Work and Engineering\01 Sensors\02 Canberra analysis"
    r"\06 Files for Python Final Round\Exhibition Environment",
    "01 Canberra Recorded Events.xlsx",
)

OUTPUT_BASE  = "Output"
# Leave as None to automatically use the LATEST "Round N" folder found in
# Output\ (so you never have to remember to update this by hand each
# round). Set to an explicit string like "Round 2" only if you want to
# (re)target a specific past round instead.
ROUND_NAME   = None
STEP1_FOLDER = "Step 1 Collation Alignment and Raw Plots"
STEP4_FOLDER = "Step 4 STL Decomposition"
STEP_FOLDER  = "Step 5 Trend Analysis"

FONT = "Arial"

SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

# 30-day settling cutoff per sensor — matches Step 1, used to filter which
# burst events are shown (a burst before a sensor's own settling period
# isn't meaningful to overlay on its trend).
SETTLING_CUTOFF = {
    "NP": "2025-06-14", "NS": "2025-06-14", "PB": "2025-06-14",
    "PT": "2025-06-14", "SB": "2025-06-14", "ST": "2025-09-27",
}

# Recorded-events category colours — taken directly from the spreadsheet's
# own cell fill colours (theme 5 / 8 / 7 respectively), not guessed.
EVENT_CATEGORY_COLOURS = {
    "weather":          "#ED7D31",   # Weather events
    "around_aircraft":  "#4472C4",   # Events around the Canberra
    "interference":     "#FFC000",   # Stepping on / direct interference
}
# Maps this workbook's theme colour index -> category. If the theme
# changes in a future export, these indices may need re-checking.
THEME_INDEX_TO_CATEGORY = {5: "weather", 8: "around_aircraft", 7: "interference"}

# Multi-day event spans (shaded regions) are detected automatically from
# the sheet itself — see load_recorded_events()'s block_id and the
# grouping logic in Plot 2: a category appearing more than once in a
# run of dated rows with no blank row between them is treated as one
# continuous span from its earliest to latest date in that run. Nothing
# to configure here; it follows whatever's in the sheet.

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


def resolve_round_name():
    """
    Returns ROUND_NAME if explicitly set, otherwise finds the
    highest-numbered "Round N" folder under Output\\ and uses that. This
    means the script picks up whichever round the full pipeline most
    recently created without needing to be edited each time.
    """
    if ROUND_NAME:
        return ROUND_NAME
    output_root = os.path.join(BASE_DIR, OUTPUT_BASE)
    if not os.path.isdir(safe_path(output_root)):
        raise FileNotFoundError(f"No Output folder found at: {output_root}")
    round_dirs = [
        d for d in os.listdir(safe_path(output_root))
        if d.startswith("Round ") and os.path.isdir(os.path.join(output_root, d))
    ]
    if not round_dirs:
        raise FileNotFoundError(f"No 'Round N' folders found under: {output_root}")

    def round_number(name):
        try:
            return int(name.replace("Round ", "").strip())
        except ValueError:
            return -1

    latest = max(round_dirs, key=round_number)
    ok(f"ROUND_NAME not set — auto-detected latest round: {latest}")
    return latest


def get_step1_dir():
    return os.path.join(BASE_DIR, OUTPUT_BASE, resolve_round_name(), STEP1_FOLDER)


def get_step4_dir():
    return os.path.join(BASE_DIR, OUTPUT_BASE, resolve_round_name(), STEP4_FOLDER)


def get_output_dir():
    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, resolve_round_name(), STEP_FOLDER)
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


def apply_month_ticks(fig, t_min, t_max):
    ticks = month_ticks(t_min, t_max)
    fig.update_xaxes(
        tickmode="array", tickvals=ticks,
        ticktext=[t.strftime("%d %b %Y") for t in ticks], tickangle=-45,
    )


def plotly_base(title, subtitle="", height=600):
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


def summarize_burst_events(df, ts_col, val_col, gap_seconds=2.0):
    """One row per distinct burst event (peak-magnitude reading) — see Step 1."""
    if df.empty:
        return df
    d = df[[ts_col, val_col]].dropna().sort_values(ts_col).reset_index(drop=True)
    gaps = d[ts_col].diff().dt.total_seconds()
    event_id = (gaps > gap_seconds).cumsum()
    idx = d.groupby(event_id)[val_col].apply(lambda s: s.abs().idxmax())
    return d.loc[idx].reset_index(drop=True)


# ── Recorded events ──────────────────────────────────────────────────────────

def load_recorded_events(path):
    """
    Reads every dated row from the recorded-events workbook, categorised
    by that row's fill colour (theme index), not by which column has
    text. Returns a list of dicts: {date, label, category, block_id}.
    block_id groups rows that have no blank row between them (see the
    block-detection logic below) — used by the caller to decide which
    events form a shaded multi-day span versus a single dashed line.
    """
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
    for row in ws.iter_rows(min_row=4):   # data starts row 4 in the known layout
        row_is_blank = all(c.value is None for c in row)
        if row_is_blank:
            if block_has_content:
                block_id += 1
                block_has_content = False
            continue
        block_has_content = True

        date_cell = row[0]   # column A
        if date_cell.value is None or not hasattr(date_cell.value, "year"):
            continue   # continuation row: text but no date of its own
        date = pd.Timestamp(date_cell.value)

        fill = date_cell.fill
        theme = fill.fgColor.theme if (fill and fill.fgColor and fill.fgColor.type == "theme") else None
        category = THEME_INDEX_TO_CATEGORY.get(theme)
        if category is None:
            continue   # structural/header-style row, not a categorised event

        # Label: combine any non-empty text across the row (Event column,
        # weather sub-label, hangar-factor note) for a fuller hover tooltip.
        texts = [str(c.value).strip() for c in row if isinstance(c.value, str) and c.value.strip()]
        label = " / ".join(texts) if texts else "(no description)"

        events.append({"date": date, "label": label, "category": category, "block_id": block_id})

    ok(f"Recorded events loaded: {len(events)} dated row(s) categorised")
    return events


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    step1_dir = get_step1_dir()
    step4_dir = get_step4_dir()
    out_dir   = get_output_dir()
    print(f"Reading Step 1 output from: {step1_dir}")
    print(f"Reading Step 4 output from: {step4_dir}")
    print(f"Output folder: {out_dir}")

    section("Loading trends")
    trends = {}
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        path = os.path.join(step4_dir, f"{sensor_key}_decomposed.csv")
        if not os.path.isfile(safe_path(path)):
            warn(f"{sensor_key}: decomposed file not found — skipped")
            continue
        df = pd.read_csv(safe_path(path), parse_dates=["Timestamp"])
        trend_col = f"{sensor_key}_Trend(\u00b5S)"
        if trend_col not in df.columns:
            warn(f"{sensor_key}: '{trend_col}' not found — skipped")
            continue
        trends[sensor_key] = dict(
            colour=colour, df=df[["Timestamp", trend_col]].dropna().reset_index(drop=True),
            trend_col=trend_col,
        )
        ok(f"{sensor_key}: {len(trends[sensor_key]['df']):,} trend points loaded")

    if not trends:
        raise SystemExit("No sensor trends loaded — run Step 4 first.")

    t_min = min(d["df"]["Timestamp"].min() for d in trends.values())
    t_max = max(d["df"]["Timestamp"].max() for d in trends.values())

    # === Plot 1: all sensors' trends, combined, no bursts/events ===
    section("Plot 1: combined trends")
    fig1 = plotly_base(
        "All Sensors - STL Trend Components (Combined)",
        "Temperature-compensated strain trend, all sensors on one chart.",
        height=680,
    )
    fig1.update_yaxes(title_text="Trend (\u00b5S)")
    for sensor_key, d in trends.items():
        label = sensor_key.replace("_", " ")
        fig1.add_trace(go.Scatter(
            x=d["df"]["Timestamp"], y=d["df"][d["trend_col"]],
            mode="lines", name=label, line=dict(color=d["colour"], width=1.0),
            connectgaps=False, showlegend=True,
            hovertemplate=f"{label}<br>%{{x|%d %b %Y}}<br>%{{y:.3f}} \u00b5S<extra></extra>",
        ))
    fig1.update_layout(legend=dict(
        orientation="h", y=-0.55, x=0, xanchor="left",
        font=dict(size=11), bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#cccccc", borderwidth=1,
    ))
    fig1.update_layout(margin=dict(l=65, r=60, t=90, b=260))
    apply_month_ticks(fig1, t_min, t_max)
    save_html(fig1, os.path.join(out_dir, "trend_all_sensors.html"))

    # === Plot 2: same combined trends + burst events + recorded events ===
    section("Plot 2: trends + bursts + recorded events")
    fig2 = plotly_base(
        "All Sensors - STL Trend Components with Burst Events and Recorded Events",
        "Bold markers = burst event (per sensor, placed on that sensor's own trend value). "
        "Shaded/vertical lines = recorded site events, coloured by category.",
        height=900,
    )
    fig2.update_yaxes(title_text="Trend (\u00b5S)")
    for sensor_key, d in trends.items():
        label = sensor_key.replace("_", " ")
        fig2.add_trace(go.Scatter(
            x=d["df"]["Timestamp"], y=d["df"][d["trend_col"]],
            mode="lines", name=label, line=dict(color=d["colour"], width=1.0),
            connectgaps=False, legendgroup=label,
            hovertemplate=f"{label}<br>%{{x|%d %b %Y}}<br>%{{y:.3f}} \u00b5S<extra></extra>",
        ))

    # ── Burst events: one marker per event, placed on that sensor's own
    #    trend value at the event's timestamp (keeps everything on the
    #    same y-scale as the trend, rather than the burst's own raw
    #    strain reading which can be a very different magnitude) ──
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        if sensor_key not in trends:
            continue
        burst_path = os.path.join(step1_dir, f"{sensor_key}_raw_burst.csv")
        if not os.path.isfile(safe_path(burst_path)):
            continue
        bdf = pd.read_csv(safe_path(burst_path), parse_dates=["Timestamp"], encoding="utf-8-sig")
        burst_col = f"{sensor_key}_Burst(\u00b5S)"
        if burst_col not in bdf.columns or bdf.empty:
            continue
        cutoff = pd.Timestamp(SETTLING_CUTOFF[folder])
        bdf = bdf[bdf["Timestamp"] >= cutoff]
        if bdf.empty:
            continue
        events = summarize_burst_events(bdf, "Timestamp", burst_col)
        if events.empty:
            continue

        trend_df = trends[sensor_key]["df"].rename(columns={trends[sensor_key]["trend_col"]: "trend"})
        merged = pd.merge_asof(
            events.sort_values("Timestamp"), trend_df.sort_values("Timestamp"),
            on="Timestamp", direction="nearest", tolerance=pd.Timedelta(hours=6),
        ).dropna(subset=["trend"])
        if merged.empty:
            continue

        label = sensor_key.replace("_", " ")
        fig2.add_trace(go.Scatter(
            x=merged["Timestamp"], y=merged["trend"],
            mode="markers", name=f"{label} (burst)",
            marker=dict(color=colour, size=9, line=dict(color="black", width=1)),
            legendgroup=label, showlegend=False,
            hovertemplate=f"{label} burst event<br>%{{x|%d %b %Y %H:%M:%S}}<extra></extra>",
        ))

    # ── Recorded events: shaded spans for multi-day events, dashed
    #    vertical lines for everything else ───────────────────────────
    # A multi-day event is any category that appears MORE THAN ONCE
    # within the same block (a run of dated rows with no blank row
    # between them, per the sheet's own convention) — the span runs
    # from the earliest to the latest date of that category within the
    # block. A category appearing only once in a block is a single-day
    # event and gets a dashed line instead. This is derived directly
    # from the sheet's structure each run, so it automatically follows
    # whatever events/dates/categories are in there — nothing to
    # maintain by hand.
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
    category_labels = {
        "weather": "Weather event", "around_aircraft": "Event around the Canberra",
        "interference": "Stepping on / interference",
    }
    for cat, colour in EVENT_CATEGORY_COLOURS.items():
        if cat in seen_categories:
            fig2.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=colour, width=2.5, dash="dash"),
                name=category_labels[cat],
            ))

    apply_month_ticks(fig2, t_min, t_max)
    fig2.update_layout(legend=dict(
        orientation="h", y=-0.37, x=0, xanchor="left",
        font=dict(size=11), bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#cccccc", borderwidth=1,
    ))
    fig2.update_layout(margin=dict(l=65, r=60, t=90, b=320))
    save_html(fig2, os.path.join(out_dir, "trend_with_bursts_and_events.html"))

    section("STEP 5 COMPLETE")
    print(f"  All outputs saved to:\n  {out_dir}")


if __name__ == "__main__":
    main()
