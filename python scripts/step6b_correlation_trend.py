"""
step6b_correlation_trend.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 6b
Inter-sensor correlation analysis on the STL TREND component from Step 4
(daily cycle and short-term noise already removed): a static Pearson
correlation matrix (with significance testing) and a 30-day rolling
Pearson correlation over time.

See also Step 6a, which runs the same analysis on compensated strain
directly instead (still includes the shared daily thermal cycle) — the
two answer different questions: 6a includes shared daily thermal response,
6b isolates longer-term load-sharing behaviour by looking at trend only.

WHAT THIS SCRIPT DOES
======================
  1. Loads all six sensors' Trend component from Step 4's combined
     output.
  2. For every sensor pair (15 pairs total), computes the Pearson
     correlation coefficient r and its two-tailed p-value (via
     scipy.stats.pearsonr) — the p-value is the "this correlation is
     unlikely to be by chance" significance test, reported the way SPSS
     conventionally does: r, p-value, and significance stars
     (*** p<0.001, ** p<0.01, * p<0.05).
  3. Plots a lower-triangle heatmap (each pair shown once, no duplicated
     values in a full square) annotated with r, stars, and p.
  4. Computes and plots a 30-day rolling Pearson correlation for every
     pair, so a shift in how sensors move together over time is visible.

OUTPUTS (saved to Output\\Round 1\\Step 6b Correlation Analysis (Trend)\\):
  correlation_table.csv       - r, p-value, significance, n for all 15 pairs
  correlation_heatmap.html    - lower-triangle heatmap
  correlation_rolling.html    - 30-day rolling Pearson r, all 15 pairs

USAGE
=====
    Run AFTER step4_stl_decomposition.py.
        python step6b_correlation_trend.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds — save
edits as step6b_correlation_trend_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import warnings
from itertools import combinations

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
STEP4_FOLDER = "Step 4 STL Decomposition"
STEP_FOLDER  = "Step 6b Correlation Analysis (Trend)"

FONT = "Arial"
STEPS_PER_DAY = 96      # 15-min intervals per day
ROLLING_DAYS  = 30

SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

PAIR_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]

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


def get_step4_dir():
    return os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP4_FOLDER)


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


def apply_month_ticks(fig, t_min, t_max):
    ticks = month_ticks(t_min, t_max)
    fig.update_xaxes(
        tickmode="array", tickvals=ticks,
        ticktext=[t.strftime("%d %b %Y") for t in ticks], tickangle=-45,
    )


def sig_stars(p):
    """SPSS-style significance stars."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    step4_dir = get_step4_dir()
    out_dir   = get_output_dir()
    print(f"Reading Step 4 output from: {step4_dir}")
    print("Correlating on: STL TREND component (daily cycle and noise removed)")
    print(f"Output folder: {out_dir}")

    master_path = os.path.join(step4_dir, "All_Sensors_Decomposed.csv")
    if not os.path.isfile(safe_path(master_path)):
        raise SystemExit(f"Could not find Step 4's combined output:\n  {master_path}")
    master = pd.read_csv(safe_path(master_path), parse_dates=["Timestamp"])
    ok(f"Loaded decomposed master file: {len(master):,} rows")

    comp_cols = {}
    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        col = f"{sensor_key}_Trend(\u00b5S)"
        if col in master.columns:
            comp_cols[sensor_key] = col
        else:
            warn(f"{sensor_key}: trend column not found — excluded from correlation")

    labels   = [sk.replace("_", " ") for sk in comp_cols]
    sk_list  = list(comp_cols.keys())
    col_list = list(comp_cols.values())
    n_sensors = len(col_list)

    if n_sensors < 2:
        raise SystemExit("Fewer than 2 sensors available — cannot compute correlation.")

    section("STEP 6b - Static Pearson correlation on TREND (lower triangle, significance-tested)")

    r_matrix = np.full((n_sensors, n_sensors), np.nan)
    p_matrix = np.full((n_sensors, n_sensors), np.nan)
    corr_rows = []

    for i in range(n_sensors):
        for j in range(n_sensors):
            if i <= j:
                continue   # diagonal + upper triangle left blank — no duplication
            mask = master[col_list[i]].notna() & master[col_list[j]].notna()
            n_pair = int(mask.sum())
            if n_pair < 3:
                continue
            si = master.loc[mask, col_list[i]].to_numpy()
            sj = master.loc[mask, col_list[j]].to_numpy()
            r_val, p_val = stats.pearsonr(si, sj)
            r_matrix[i, j] = r_val
            p_matrix[i, j] = p_val
            corr_rows.append({
                "Sensor A": labels[i], "Sensor B": labels[j],
                "Pearson r": round(float(r_val), 4),
                "p-value": float(p_val),
                "Significance": sig_stars(p_val),
                "n": n_pair,
            })
            ok(f"{labels[i]} vs {labels[j]}: r={r_val:.4f}{sig_stars(p_val)}  p={p_val:.4g}  n={n_pair:,}")

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(
        safe_path(os.path.join(out_dir, "correlation_table.csv")),
        index=False, encoding="utf-8-sig",
    )
    ok("correlation_table.csv saved")

    # ── Lower-triangle heatmap ──────────────────────────────────────────
    annotations = []
    hover_text = []
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
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=f"<b>{r_val:.2f}{stars}</b><br><span style='font-size:8px'>{p_text}</span>",
                font=dict(size=11, family=FONT, color=text_colour),
                showarrow=False, align="center",
            ))
            n_pair = int((master[col_list[i]].notna() & master[col_list[j]].notna()).sum())
            p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"
            row_hover.append(
                f"<b>{labels[i]}</b> vs <b>{labels[j]}</b><br>"
                f"r = {r_val:.4f}{stars}<br>p {p_str}<br>n = {n_pair:,}"
            )
        hover_text.append(row_hover)

    fig_heat = go.Figure(go.Heatmap(
        z=r_matrix, x=labels, y=labels,
        text=hover_text, hovertemplate="%{text}<extra></extra>",
        colorscale=[
            [0.0, "#1565C0"], [0.25, "#90CAF9"], [0.5, "#FFFFFF"],
            [0.75, "#EF9A9A"], [1.0, "#B71C1C"],
        ],
        zmin=-1, zmax=1,
        colorbar=dict(title=dict(text="Pearson r", side="right"),
                      tickvals=[-1, -0.5, 0, 0.5, 1], thickness=16),
    ))
    fig_heat.update_layout(
        title=dict(
            text=("Inter-Sensor Pearson Correlation (Trend) - Lower Triangle Heatmap<br>"
                  "<sup>STL trend component (daily cycle removed), full monitoring period. "
                  "Significance: *** p&lt;0.001  ** p&lt;0.01  * p&lt;0.05  |  "
                  "Red = positive, Blue = negative, White = no correlation.</sup>"),
            font=dict(size=13, family=FONT), x=0.0, xanchor="left",
        ),
        annotations=annotations,
        xaxis=dict(tickfont=dict(size=10), side="bottom"),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=600, width=720,
        margin=dict(l=150, r=80, t=110, b=130),
        font=dict(family=FONT),
    )
    save_html(fig_heat, os.path.join(out_dir, "correlation_heatmap.html"))

    # ── 30-day rolling Pearson correlation ──────────────────────────────
    section("30-day rolling Pearson correlation")
    rolling_steps = ROLLING_DAYS * STEPS_PER_DAY
    pairs = list(combinations(range(n_sensors), 2))

    fig_roll = go.Figure()
    fig_roll.update_layout(
        title=dict(
            text=(f"Rolling Pearson Correlation ({ROLLING_DAYS}-day window, Trend) - All Sensor Pairs<br>"
                  "<sup>STL trend component (daily cycle removed). Stable lines = consistent load sharing. "
                  "Sudden shifts = potential structural change.</sup>"),
            font=dict(size=13, family=FONT), x=0.0, xanchor="left",
        ),
        xaxis=dict(showgrid=True, gridcolor="#eeeeee", type="date",
                    rangeslider=dict(visible=True, thickness=0.035)),
        yaxis=dict(title="Pearson r", range=[-1.05, 1.05],
                    tickvals=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
                    showgrid=True, gridcolor="#eeeeee"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.55, x=0, xanchor="left",
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#cccccc", borderwidth=1, tracegroupgap=4),
        height=820,
        margin=dict(l=65, r=60, t=100, b=320),
        hovermode="x unified",
        font=dict(family=FONT),
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
        si = master[col_list[i]].reset_index(drop=True)
        sj = master[col_list[j]].reset_index(drop=True)
        roll_r = si.rolling(window=rolling_steps, min_periods=rolling_steps // 2).corr(sj)
        fig_roll.add_trace(go.Scatter(
            x=master["Timestamp"], y=roll_r.values,
            mode="lines", name=pair_name,
            line=dict(color=PAIR_COLOURS[k % len(PAIR_COLOURS)], width=1.0),
            connectgaps=False,
            hovertemplate=f"<b>{pair_name}</b><br>%{{x|%d %b %Y}}<br>r = %{{y:.3f}}<extra></extra>",
        ))

    t_min, t_max = master["Timestamp"].min(), master["Timestamp"].max()
    apply_month_ticks(fig_roll, t_min, t_max)
    save_html(fig_roll, os.path.join(out_dir, "correlation_rolling.html"))

    section("STEP 6b COMPLETE")
    print(f"  All outputs saved to:\n  {out_dir}")


if __name__ == "__main__":
    main()
