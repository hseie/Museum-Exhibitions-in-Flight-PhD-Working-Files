"""
step7_trend_model_fitting.py
------------------------------------------------------------------------------
Canberra PR9 (XH171) SHM Project — Step 7
Trend model fitting: for each sensor's daily-median compensated strain,
fit several candidate long-term models and select the best by BIC (AIC
as tie-break).

WHAT THIS SCRIPT DOES
======================
For each of the six strain sensors:
  1. Takes the daily median of Step 3's compensated strain (smooths out
     the daily cycle and short-term noise without removing information
     the way STL's trend extraction does).
  2. Fits several candidate models against days-since-that-sensor's-own-
     start:
       - Polynomial, degrees 1-5
       - Damped sinusoid: A * exp(-gamma*t) * sin(omega*t + phi) + C
       - Piecewise linear with one breakpoint (detected via `ruptures`)
     Plus LOWESS as a visual-only reference (not compared by AIC/BIC —
     it isn't a parametric model with a comparable likelihood).
  3. Computes AIC and BIC for every candidate; selects the best by BIC
     (simpler models preferred when AIC and BIC disagree), and reports
     which selection rule decided it.
  4. Plots all candidates (faded) with the best fit highlighted (solid
     black, starred in the legend) and LOWESS shown as a dotted
     reference line.

OUTPUTS (saved to Output\\Round 1\\Step 7 Trend Model Fitting\\):
  aic_bic_summary.csv         - every candidate's k, AIC, BIC, R2, and
                                 whether it was selected, per sensor
  model_fit_<Sensor>.html     - daily-median points, all candidates faded,
                                 best fit highlighted, LOWESS reference (x6)

USAGE
=====
    Run AFTER step3_temperature_compensation.py.
        python step7_trend_model_fitting.py

SCRIPT VERSIONING
=================
Per project convention: never overwrite this script between rounds — save
edits as step7_trend_model_fitting_v2.py, _v3.py, etc.
------------------------------------------------------------------------------
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import ruptures
from statsmodels.nonparametric.smoothers_lowess import lowess
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
STEP3_FOLDER = "Step 3 Compensation"
STEP_FOLDER  = "Step 7 Trend Model Fitting"

FONT = "Arial"

SENSOR_MAP = {
    "NP": ("Nose_Portside",    "#2196F3"),
    "NS": ("Nose_Starboard",   "#FF9800"),
    "PB": ("Portside_Bottom",  "#4CAF50"),
    "PT": ("Portside_Top",     "#9C27B0"),
    "SB": ("Starboard_Bottom", "#F44336"),
    "ST": ("Starboard_Top",    "#8B4513"),
}

NON_BEST_COLOURS = [
    "#A5D6A7", "#90CAF9", "#FFCC80", "#CE93D8",
    "#F48FB1", "#80DEEA", "#BCAAA4", "#EF9A9A",
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


def get_step3_dir():
    return os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP3_FOLDER)


def get_output_dir():
    out_dir = os.path.join(BASE_DIR, OUTPUT_BASE, ROUND_NAME, STEP_FOLDER)
    os.makedirs(safe_path(out_dir), exist_ok=True)
    return out_dir


def save_html(fig, path):
    fig.write_html(safe_path(path), include_plotlyjs="cdn")
    ok(f"Saved: {os.path.basename(path)}")


def compute_aic_bic(n, k, rss):
    if n <= k or rss <= 0:
        return np.inf, np.inf
    sigma2  = rss / n
    log_lik = -n / 2 * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
    return 2 * k - 2 * log_lik, k * np.log(n) - 2 * log_lik


def damped_sinusoid(t, A, omega, phi, gamma, C):
    return A * np.exp(-gamma * t) * np.sin(omega * t + phi) + C


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    step3_dir = get_step3_dir()
    out_dir   = get_output_dir()
    print(f"Reading Step 3 output from: {step3_dir}")
    print(f"Output folder: {out_dir}")

    section("STEP 7 - Trend model fitting (daily median, AIC/BIC selection)")

    all_model_rows = []

    for folder, (sensor_key, colour) in SENSOR_MAP.items():
        comp_path = os.path.join(step3_dir, f"{sensor_key}_compensated.csv")
        if not os.path.isfile(safe_path(comp_path)):
            warn(f"{sensor_key}: compensated file not found — skipped")
            continue

        df = pd.read_csv(safe_path(comp_path), parse_dates=["Timestamp"])
        comp_col = f"{sensor_key}_Compensated(\u00b5S)"
        if comp_col not in df.columns:
            warn(f"{sensor_key}: '{comp_col}' not found — skipped")
            continue

        label = sensor_key.replace("_", " ")
        df["Date"] = df["Timestamp"].dt.normalize()
        daily = df.groupby("Date")[comp_col].median().reset_index()
        daily.columns = ["Date", "y"]
        daily = daily.dropna()
        if len(daily) < 10:
            warn(f"{sensor_key}: insufficient daily rows ({len(daily)}) — skipped")
            continue

        t0 = daily["Date"].min()
        t  = ((daily["Date"] - t0).dt.total_seconds() / 86400).to_numpy()
        y  = daily["y"].to_numpy()
        n  = len(y)

        candidates = []

        # Polynomials, degrees 1-5
        for deg in range(1, 6):
            X = np.column_stack([t ** d for d in range(deg + 1)])
            k = deg + 1
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                y_hat  = X @ coeffs
                rss    = float(np.sum((y - y_hat) ** 2))
                aic, bic = compute_aic_bic(n, k, rss)
                r2 = 1 - rss / np.sum((y - y.mean()) ** 2)
                candidates.append(dict(name=f"Polynomial deg {deg}", k=k,
                                        aic=aic, bic=bic, r2=r2, y_hat=y_hat))
            except Exception:
                pass

        # Damped sinusoid
        try:
            period_guess = 365.0
            p0 = [y.std(), 2 * np.pi / period_guess, 0, 1e-4, y.mean()]
            popt, _ = curve_fit(
                damped_sinusoid, t, y, p0=p0, maxfev=10000,
                bounds=([-np.inf, 1e-6, -np.pi, 0, -np.inf],
                        [np.inf, np.inf, np.pi, np.inf, np.inf]),
            )
            y_hat = damped_sinusoid(t, *popt)
            rss   = float(np.sum((y - y_hat) ** 2))
            k     = 5
            aic, bic = compute_aic_bic(n, k, rss)
            r2 = 1 - rss / np.sum((y - y.mean()) ** 2)
            candidates.append(dict(name="Damped sinusoid", k=k,
                                    aic=aic, bic=bic, r2=r2, y_hat=y_hat))
        except Exception as e:
            warn(f"{sensor_key}: damped sinusoid fit failed ({e})")

        # Piecewise linear, one breakpoint
        try:
            algo  = ruptures.Dynp(model="l2").fit(y.reshape(-1, 1))
            bkpts = algo.predict(n_bkps=1)
            bp    = bkpts[0]
            X_pw  = np.zeros((n, 4))
            X_pw[:, 0] = 1
            X_pw[:, 1] = t
            X_pw[:, 2] = (t > t[bp]).astype(float) if bp < n else 0
            X_pw[:, 3] = np.where(t > t[bp], t - t[bp], 0) if bp < n else 0
            k = 4
            coeffs_pw = np.linalg.lstsq(X_pw, y, rcond=None)[0]
            y_hat = X_pw @ coeffs_pw
            rss   = float(np.sum((y - y_hat) ** 2))
            aic, bic = compute_aic_bic(n, k, rss)
            r2 = 1 - rss / np.sum((y - y.mean()) ** 2)
            candidates.append(dict(name="Piecewise linear", k=k,
                                    aic=aic, bic=bic, r2=r2, y_hat=y_hat,
                                    bp_date=daily["Date"].iloc[min(bp, n - 1)]))
        except Exception as e:
            warn(f"{sensor_key}: piecewise linear fit failed ({e})")

        if not candidates:
            warn(f"{sensor_key}: no models fitted — skipped")
            continue

        # LOWESS — visual reference only, not compared via AIC/BIC
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

        ok(f"{sensor_key}: best={best['name']}  AIC={best['aic']:.1f}  "
           f"BIC={best['bic']:.1f}  R2={best['r2']:.4f}  [{sel_note}]")

        for c in candidates:
            all_model_rows.append({
                "Sensor": label, "Model": c["name"], "k": c["k"],
                "AIC": round(c["aic"], 2), "BIC": round(c["bic"], 2),
                "R2": round(c["r2"], 4),
                "Selected": "YES" if c["name"] == best["name"] else "",
                "Note": sel_note if c["name"] == best["name"] else "",
            })

        # ── Model fit plot ───────────────────────────────────────────────
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["Date"], y=y, mode="markers",
            marker=dict(color=colour, size=3, opacity=0.5),
            name="Daily median",
            hovertemplate="%{x|%d %b %Y}<br>%{y:.3f} \u00b5S<extra></extra>",
        ))
        colour_idx = 0
        for c in candidates:
            if c["name"] == best["name"]:
                continue
            fig.add_trace(go.Scatter(
                x=daily["Date"], y=c["y_hat"], mode="lines",
                name=f"{c['name']}  (AIC={c['aic']:.0f})",
                line=dict(color=NON_BEST_COLOURS[colour_idx % len(NON_BEST_COLOURS)],
                           width=1.0, dash="dot"),
                opacity=0.55,
                hovertemplate=f"{c['name']}<br>%{{x|%d %b %Y}}<br>%{{y:.3f}} \u00b5S<extra></extra>",
            ))
            colour_idx += 1
        if lowess_y is not None:
            fig.add_trace(go.Scatter(
                x=daily["Date"], y=lowess_y, mode="lines",
                name="LOWESS (visual reference)",
                line=dict(color="#607D8B", width=1.2, dash="dot"), opacity=0.7,
            ))
        fig.add_trace(go.Scatter(
            x=daily["Date"], y=best["y_hat"], mode="lines",
            name=f"\u2605 Best: {best['name']}  (AIC={best['aic']:.0f}, R\u00b2={best['r2']:.4f})",
            line=dict(color="#111111", width=2.2),
            hovertemplate=f"Best fit: {best['name']}<br>%{{x|%d %b %Y}}<br>%{{y:.3f}} \u00b5S<extra></extra>",
        ))
        if "bp_date" in best:
            fig.add_vline(
                x=best["bp_date"].timestamp() * 1000,
                line_dash="dot", line_color="#999999",
                annotation_text="Breakpoint", annotation_font_size=10,
                annotation_position="top right",
            )

        fig.update_layout(
            title=dict(
                text=(f"{label} - Trend Model Fitting (daily median)<br>"
                      f"<sup>\u2605 Best: {best['name']}  |  R\u00b2={best['r2']:.4f}  |  "
                      f"AIC={best['aic']:.1f}  |  BIC={best['bic']:.1f}  |  {sel_note}</sup>"),
                font=dict(size=12, family=FONT), x=0.0, xanchor="left",
            ),
            xaxis=dict(title="Date", showgrid=True, gridcolor="#eeeeee",
                        rangeslider=dict(visible=True, thickness=0.04), type="date"),
            yaxis=dict(title="Compensated Strain (\u00b5S)", showgrid=True, gridcolor="#eeeeee"),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.38, x=0, xanchor="left",
                        font=dict(size=10), bgcolor="rgba(255,255,255,0.95)",
                        bordercolor="#cccccc", borderwidth=1),
            height=680,
            margin=dict(l=65, r=50, t=100, b=190),
            font=dict(family=FONT),
        )
        save_html(fig, os.path.join(out_dir, f"model_fit_{sensor_key}.html"))

    if not all_model_rows:
        raise SystemExit("No sensor produced model fits — nothing to save.")

    pd.DataFrame(all_model_rows).to_csv(
        safe_path(os.path.join(out_dir, "aic_bic_summary.csv")),
        index=False, encoding="utf-8-sig",
    )
    ok("aic_bic_summary.csv saved")

    section("STEP 7 COMPLETE")
    print(f"  All outputs saved to:\n  {out_dir}")


if __name__ == "__main__":
    main()
