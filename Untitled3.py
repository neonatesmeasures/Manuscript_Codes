#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHD <-> CBC analysis — two complementary directions:

  Direction 1 (original): CBC trajectory features (weeks 0–12) → predict CHD
  Direction 2 (new):      CHD status → WBC values at 15–26 weeks
                          (group comparison + linear regression)

Run:
  python run_chd_metrics_v2.py --base-dir "<PATH_TO_DATA_DIR>"
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

try:
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, roc_curve,
        precision_recall_curve, f1_score, confusion_matrix
    )
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except ImportError as e:
    raise SystemExit(
        "Missing dependencies. Install with:\n"
        "  pip install scikit-learn statsmodels scipy matplotlib pandas numpy\n"
        f"Details: {e}"
    )

# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def extract_week_number(colname: str) -> Optional[int]:
    m = re.search(r"(\d+)(?!.*\d)", colname)
    return int(m.group(1)) if m else None


def week_columns_for_measurement(df: pd.DataFrame, measurement_name: str) -> Dict[int, str]:
    patt = re.escape(measurement_name)
    cols = [c for c in df.columns if re.search(patt, c, flags=re.IGNORECASE)]
    wmap = {}
    for c in cols:
        w = extract_week_number(c)
        if w is not None and 0 <= w <= 26:
            wmap[w] = c
    return dict(sorted(wmap.items()))


def compute_week_reference(df: pd.DataFrame, week_cols: Dict[int, str]) -> pd.DataFrame:
    rows = []
    for w, col in week_cols.items():
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "week": w,
            "mean": float(vals.mean()) if len(vals) else np.nan,
            "std":  float(vals.std(ddof=0)) if len(vals) else np.nan,
        })
    ref = pd.DataFrame(rows).sort_values("week")
    ref["mean"] = ref["mean"].interpolate(limit_direction="both")
    ref["std"]  = ref["std"].replace(0, np.nan).interpolate(limit_direction="both")
    if ref["std"].isna().any():
        ref["std"] = ref["std"].fillna(ref["std"].median())
    return ref


def add_zscores(df: pd.DataFrame, week_cols: Dict[int, str], ref: pd.DataFrame) -> pd.DataFrame:
    for w, col in week_cols.items():
        mu_s = ref.loc[ref["week"] == w, "mean"]
        sd_s = ref.loc[ref["week"] == w, "std"]
        if mu_s.empty or sd_s.empty:
            continue
        mu = float(mu_s.values[0])
        sd = float(sd_s.values[0]) if float(sd_s.values[0]) != 0 else 1.0
        df[f"{col}__z"] = (pd.to_numeric(df[col], errors="coerce") - mu) / sd
    return df


def summarize_series(values: List[float]) -> Tuple[float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    if len(arr) == 0:
        return (np.nan, np.nan)
    meanv = float(np.mean(arr))
    slope = np.nan
    if len(arr) >= 3:
        x = np.arange(len(arr))
        slope = float(np.polyfit(x, arr, 1)[0])
    return (meanv, slope)


def rolling_flag(values: List[float], thresh_abs: float, needed_weeks: int, mode: str = ">=") -> int:
    ser = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if mode.strip() == "<=":
        arr = ser.fillna(float("inf")).abs().values
        cmp = lambda v: v <= thresh_abs
    else:
        arr = ser.fillna(0.0).abs().values
        cmp = lambda v: v >= thresh_abs
    run = 0
    for v in arr:
        if cmp(v):
            run += 1
            if run >= needed_weeks:
                return 1
        else:
            run = 0
    return 0


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = (2 * prec * rec) / np.clip(prec + rec, 1e-12, None)
    f1s = f1s[1:]
    if len(f1s) == 0:
        return 0.5, 0.0
    idx = int(np.argmax(f1s))
    return float(thr[idx]), float(f1s[idx])


# ─────────────────────────────────────────────
# Direction 1: CBC (0–12w) → CHD
# ─────────────────────────────────────────────

def run_direction1(df: pd.DataFrame, MEASUREMENT_NAMES: List[str],
                   PRE_DIAG_MAX_WEEK: int, PERSON_ID_COL: str) -> dict:
    """
    Builds per-person trajectory features from weeks 0–PRE_DIAG_MAX_WEEK,
    fits logistic regression, returns metrics dict and feature DataFrame.
    """
    print("\n" + "="*60)
    print("DIRECTION 1: CBC trajectory (0–12w) → CHD prediction")
    print("="*60)

    feature_frames = []
    for meas in MEASUREMENT_NAMES:
        wcols = week_columns_for_measurement(df, meas)
        if not wcols:
            print(f"  [WARN] No week columns for: {meas}")
            continue

        ref = compute_week_reference(df, wcols)
        df  = add_zscores(df, wcols, ref)

        zcols = {w: f"{col}__z" for w, col in wcols.items()
                 if w <= PRE_DIAG_MAX_WEEK and f"{col}__z" in df.columns}
        if not zcols:
            continue

        ordered_weeks = sorted(zcols.keys())
        zlist = [zcols[w] for w in ordered_weeks]
        tag   = re.sub(r"[^A-Za-z]+", "_", meas).strip("_").lower()

        means, slopes, devs, convs, rerise = [], [], [], [], []
        for _, row in df.iterrows():
            vals = [row.get(c, np.nan) for c in zlist]
            meanv, slopev = summarize_series(vals)
            means.append(meanv)
            slopes.append(slopev)
            devs.append(rolling_flag(vals, 1.5, 2))
            convs.append(rolling_flag(vals, 1.0, 3, mode="<="))
            if "neutrophil" in meas.lower():
                sub = [float(row[zcols[w]]) for w in range(12, 16)
                       if w in zcols and pd.notna(row.get(zcols[w], np.nan))]
                if len(sub) >= 3:
                    s = float(np.polyfit(np.arange(len(sub)), sub, 1)[0])
                    rerise.append(1 if s <= 0 else 0)
                else:
                    rerise.append(0)
            else:
                rerise.append(0)

        feat = pd.DataFrame({
            PERSON_ID_COL:                   df[PERSON_ID_COL].values,
            f"{tag}_mean_0_{PRE_DIAG_MAX_WEEK}w":  means,
            f"{tag}_slope_0_{PRE_DIAG_MAX_WEEK}w": slopes,
            f"{tag}_deviation_flag":         devs,
            f"{tag}_convergence_flag":       convs,
            f"{tag}_re_rise_alert":          rerise,
        })
        feature_frames.append(feat)

    if not feature_frames:
        raise SystemExit("No features — check measurement names.")

    features = feature_frames[0]
    for k in feature_frames[1:]:
        features = features.merge(k, on=PERSON_ID_COL, how="outer")
    features = features.merge(df[[PERSON_ID_COL, "CHD"]], on=PERSON_ID_COL, how="inner")
    if "gestational_age" in df.columns:
        features = features.merge(df[[PERSON_ID_COL, "gestational_age"]],
                                  on=PERSON_ID_COL, how="left")

    features.to_csv("chd_features_0_12w.csv", index=False)

    drop_cols = {PERSON_ID_COL, "CHD"}
    X_cols = [c for c in features.columns if c not in drop_cols and features[c].dtype != "O"]
    X = features[X_cols].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = features["CHD"].astype(int)
    valid = ~X.isna().all(axis=1)
    X, y = X.loc[valid], y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)

    model = LogisticRegression(max_iter=5000, solver="lbfgs")
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    auroc      = roc_auc_score(y_test, probs)
    auprc      = average_precision_score(y_test, probs)
    prevalence = float(y_test.mean())

    fpr, tpr, _ = roc_curve(y_test, probs)
    prec, rec, _ = precision_recall_curve(y_test, probs)

    # Save curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#2166ac", lw=2, label=f"AUROC = {auroc:.3f}")
    axes[0].plot([0,1],[0,1], "--", color="grey", lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC curve — CBC (0–12w) → CHD"); axes[0].legend()

    axes[1].plot(rec, prec, color="#d6604d", lw=2, label=f"AUPRC = {auprc:.3f}")
    axes[1].axhline(prevalence, ls="--", color="grey", lw=1,
                    label=f"Baseline (prev={prevalence:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("PR curve — CBC (0–12w) → CHD"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("dir1_roc_pr_curves.png", dpi=160)
    plt.close()

    best_thr, best_f1 = best_f1_threshold(y_test.values, probs)
    y_pred = (probs >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "direction": "1 — CBC (0–12w) → CHD",
        "test_size": int(len(y_test)),
        "prevalence": prevalence,
        "AUROC": round(auroc, 4),
        "AUPRC": round(auprc, 4),
        "AUPRC_baseline": round(prevalence, 4),
        "best_F1_threshold": round(best_thr, 4),
        "best_F1": round(best_f1, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    print(f"  Test n={len(y_test)} | prevalence={prevalence:.3f}")
    print(f"  AUROC = {auroc:.3f} | AUPRC = {auprc:.3f}")
    print(f"  Best-F1 @ {best_thr:.3f} → F1={best_f1:.3f}  (TP={tp}, FP={fp}, FN={fn}, TN={tn})")

    return metrics, features


# ─────────────────────────────────────────────
# Direction 2 (NEW): CHD → WBC values 15–26w
# ─────────────────────────────────────────────

def run_direction2(df: pd.DataFrame, MEASUREMENT_NAMES: List[str],
                   PERSON_ID_COL: str, has_ga: bool) -> dict:
    """
    For each CBC cell type:
      1. Compute per-person mean value across weeks 15–26
      2. Mann–Whitney U test  (CHD vs non-CHD)
      3. OLS regression: mean_wbc ~ CHD [+ gestational_age]
      4. Trajectory plot showing group means ± SE across weeks 15–26

    Addresses the reviewer's request to assess whether CHD predicts
    actual WBC values rather than the probability of being tested.
    """
    print("\n" + "="*60)
    print("DIRECTION 2 (NEW): CHD status → WBC values at 15–26 weeks")
    print("="*60)

    WINDOW = (15, 26)
    results = {}
    all_means = []

    for meas in MEASUREMENT_NAMES:
        wcols = week_columns_for_measurement(df, meas)
        window_wcols = {w: c for w, c in wcols.items()
                        if WINDOW[0] <= w <= WINDOW[1]}
        if not window_wcols:
            print(f"  [WARN] No 15–26w columns for: {meas}")
            continue

        short = meas.split("/")[0].strip()  # e.g. "Neutrophils"
        tag   = re.sub(r"[^A-Za-z]+", "_", meas).strip("_").lower()

        # --- per-person mean across weeks 15–26 ---
        numeric_cols = [window_wcols[w] for w in sorted(window_wcols)]
        sub = df[[PERSON_ID_COL, "CHD"] + numeric_cols].copy()
        for c in numeric_cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

        sub[f"{tag}_mean_15_26w"] = sub[numeric_cols].mean(axis=1, skipna=True)
        sub = sub.dropna(subset=[f"{tag}_mean_15_26w"])
        if has_ga and "gestational_age" in df.columns:
            sub = sub.merge(df[[PERSON_ID_COL, "gestational_age"]],
                            on=PERSON_ID_COL, how="left")

        chd_vals    = sub.loc[sub["CHD"] == 1, f"{tag}_mean_15_26w"].values
        nonchd_vals = sub.loc[sub["CHD"] == 0, f"{tag}_mean_15_26w"].values

        if len(chd_vals) < 5 or len(nonchd_vals) < 5:
            print(f"  [SKIP] Too few observations for {short}")
            continue

        # --- Mann–Whitney U ---
        u_stat, p_mw = stats.mannwhitneyu(chd_vals, nonchd_vals, alternative="two-sided")
        # Effect size: rank-biserial correlation
        n1, n2 = len(chd_vals), len(nonchd_vals)
        rb_r = 1 - (2 * u_stat) / (n1 * n2)

        # --- OLS regression: mean_wbc ~ CHD [+ GA] ---
        formula_base = f"{tag}_mean_15_26w ~ CHD"
        if has_ga and "gestational_age" in sub.columns and sub["gestational_age"].notna().sum() > 10:
            formula = formula_base + " + gestational_age"
        else:
            formula = formula_base

        ols_sub = sub[[f"{tag}_mean_15_26w", "CHD"] +
                       (["gestational_age"] if "gestational_age" in sub.columns else [])
                      ].dropna()

        try:
            ols_res = smf.ols(formula, data=ols_sub).fit()
            beta_chd  = float(ols_res.params.get("CHD", np.nan))
            se_chd    = float(ols_res.bse.get("CHD", np.nan))
            pval_ols  = float(ols_res.pvalues.get("CHD", np.nan))
            r2        = float(ols_res.rsquared)
        except Exception as ex:
            print(f"  [WARN] OLS failed for {short}: {ex}")
            beta_chd = se_chd = pval_ols = r2 = np.nan

        results[short] = {
            "n_CHD":         int(n1),
            "n_nonCHD":      int(n2),
            "CHD_mean":      round(float(np.mean(chd_vals)), 3),
            "CHD_sd":        round(float(np.std(chd_vals, ddof=1)), 3),
            "nonCHD_mean":   round(float(np.mean(nonchd_vals)), 3),
            "nonCHD_sd":     round(float(np.std(nonchd_vals, ddof=1)), 3),
            "MannWhitney_U": round(float(u_stat), 1),
            "MannWhitney_p": float(f"{p_mw:.4e}"),
            "rank_biserial_r": round(rb_r, 3),
            "OLS_beta_CHD":  round(beta_chd, 4) if not np.isnan(beta_chd) else None,
            "OLS_SE":        round(se_chd, 4)   if not np.isnan(se_chd)   else None,
            "OLS_p_CHD":     float(f"{pval_ols:.4e}") if not np.isnan(pval_ols) else None,
            "OLS_R2":        round(r2, 4)        if not np.isnan(r2)        else None,
            "formula":       formula,
        }

        print(f"\n  [{short}]")
        print(f"    CHD   : n={n1}  mean={np.mean(chd_vals):.2f}  SD={np.std(chd_vals,ddof=1):.2f}")
        print(f"    non-CHD: n={n2}  mean={np.mean(nonchd_vals):.2f}  SD={np.std(nonchd_vals,ddof=1):.2f}")
        print(f"    Mann-Whitney U={u_stat:.1f}  p={p_mw:.4e}  rank-biserial r={rb_r:.3f}")
        print(f"    OLS β(CHD)={beta_chd:.4f}  SE={se_chd:.4f}  p={pval_ols:.4e}  R²={r2:.4f}")

        # Weekly trajectory data for plotting
        for w, col in sorted(window_wcols.items()):
            chd_w    = pd.to_numeric(df.loc[df["CHD"]==1, col], errors="coerce").dropna().values
            nonchd_w = pd.to_numeric(df.loc[df["CHD"]==0, col], errors="coerce").dropna().values
            for grp, arr in [("CHD", chd_w), ("non-CHD", nonchd_w)]:
                if len(arr) > 0:
                    all_means.append({
                        "cell_type": short,
                        "week": w,
                        "group": grp,
                        "mean": float(np.mean(arr)),
                        "se":   float(stats.sem(arr)),
                        "n":    len(arr),
                    })

    # ---- Trajectory plot (weekly means ± SE, 15–26w) ----
    if all_means:
        traj = pd.DataFrame(all_means)
        cell_types = traj["cell_type"].unique()
        n_ct  = len(cell_types)
        ncols = min(3, n_ct)
        nrows = math.ceil(n_ct / ncols)

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4.2 * nrows),
                                 squeeze=False)
        colors = {"CHD": "#d6604d", "non-CHD": "#4393c3"}

        for idx, ct in enumerate(cell_types):
            ax = axes[idx // ncols][idx % ncols]
            sub_t = traj[traj["cell_type"] == ct]
            for grp, grp_df in sub_t.groupby("group"):
                gd = grp_df.sort_values("week")
                ax.plot(gd["week"], gd["mean"], "o-", color=colors[grp],
                        lw=2, label=grp, markersize=5)
                ax.fill_between(gd["week"],
                                gd["mean"] - gd["se"],
                                gd["mean"] + gd["se"],
                                alpha=0.20, color=colors[grp])
            ax.set_title(ct, fontsize=12, fontweight="bold")
            ax.set_xlabel("Postnatal week")
            ax.set_ylabel("% of leukocytes (mean ± SE)")
            ax.legend(fontsize=9)
            ax.set_xlim(WINDOW[0] - 0.5, WINDOW[1] + 0.5)

        # Hide unused axes
        for idx in range(n_ct, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(
            "WBC subtype trajectories (weeks 15–26): CHD vs non-CHD\n"
            "(Direction 2: CHD status → WBC values)",
            fontsize=13, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        plt.savefig("dir2_wbc_trajectories_15_26w.png", dpi=160, bbox_inches="tight")
        plt.close()
        print("\n  [OK] Saved: dir2_wbc_trajectories_15_26w.png")

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--pre-diag-max-week", type=int, default=12)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    pivot_path  = base_dir / "babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv"
    labels_path = base_dir / "congenital_circ_labels.csv"
    ga_path     = base_dir / "Babies_GA.csv"

    for p in [pivot_path, labels_path]:
        if not p.exists():
            raise SystemExit(f"Required file missing: {p}")

    pivot  = pd.read_csv(pivot_path)
    labels = pd.read_csv(labels_path)
    ga     = pd.read_csv(ga_path) if ga_path.exists() else None

    PERSON_ID_COL = "person_id"

    def find_chd_label_column(ldf):
        candidates = [c for c in ldf.columns if c != PERSON_ID_COL]
        priority   = [c for c in candidates
                      if any(k in c.lower() for k in ["chd","cong","circ","card","heart"])]
        for col in priority + [c for c in candidates if c not in priority]:
            vals = pd.to_numeric(ldf[col], errors="coerce").dropna().unique()
            if set(vals).issubset({0, 1}):
                return col
        raise SystemExit("Cannot auto-detect CHD label column.")

    label_col = find_chd_label_column(labels)
    labels = (labels[[PERSON_ID_COL, label_col]]
              .rename(columns={label_col: "CHD"}))
    labels["CHD"] = pd.to_numeric(labels["CHD"], errors="coerce").fillna(0).astype(int)

    has_ga = False
    if ga is not None and PERSON_ID_COL in ga.columns:
        ga_col = next((c for c in ga.columns
                       if c != PERSON_ID_COL and ("gest" in c.lower() or c.lower()=="ga")),
                      None)
        if ga_col is None and len([c for c in ga.columns if c!=PERSON_ID_COL]) == 1:
            ga_col = [c for c in ga.columns if c != PERSON_ID_COL][0]
        if ga_col:
            ga = ga[[PERSON_ID_COL, ga_col]].rename(columns={ga_col: "gestational_age"})
            has_ga = True

    df = pivot.merge(labels, on=PERSON_ID_COL, how="inner")
    if has_ga:
        df = df.merge(ga, on=PERSON_ID_COL, how="left")
    print(f"[OK] Merged dataset: {df.shape[0]:,} infants, {df.shape[1]} columns")
    print(f"     CHD cases: {df['CHD'].sum()} ({df['CHD'].mean()*100:.1f}%)")

    MEASUREMENT_NAMES = [
        "Neutrophils/100 leukocytes in Blood",
        "Lymphocytes/100 leukocytes in Blood",
        "Monocytes/100 leukocytes in Blood",
        "Eosinophils/100 leukocytes in Blood",
        "Basophils/100 leukocytes in Blood",
    ]

    # ── Run both directions ──────────────────────
    dir1_metrics, features = run_direction1(
        df, MEASUREMENT_NAMES, int(args.pre_diag_max_week), PERSON_ID_COL)

    dir2_results = run_direction2(
        df, MEASUREMENT_NAMES, PERSON_ID_COL, has_ga)

    # ── Save consolidated JSON ───────────────────
    summary = {
        "direction_1": dir1_metrics,
        "direction_2_CHD_predicts_WBC_15_26w": dir2_results,
    }
    with open("metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Summary table for Direction 2 ────────────
    if dir2_results:
        rows = []
        for cell, r in dir2_results.items():
            rows.append({
                "Cell type":       cell,
                "CHD mean (SD)":   f"{r['CHD_mean']} ({r['CHD_sd']})",
                "non-CHD mean (SD)": f"{r['nonCHD_mean']} ({r['nonCHD_sd']})",
                "MW p-value":      r["MannWhitney_p"],
                "rank-biserial r": r["rank_biserial_r"],
                "OLS β (CHD)":     r["OLS_beta_CHD"],
                "OLS p":           r["OLS_p_CHD"],
                "R²":              r["OLS_R2"],
            })
        table = pd.DataFrame(rows)
        table.to_csv("dir2_group_comparison_table.csv", index=False)
        print("\n[OK] Saved: dir2_group_comparison_table.csv")
        print("\nDirection 2 — summary table:")
        print(table.to_string(index=False))

    print("\n" + "="*60)
    print("ALL OUTPUTS")
    print("="*60)
    print("  chd_features_0_12w.csv          — Direction 1 feature matrix")
    print("  dir1_roc_pr_curves.png           — Direction 1 ROC + PR curves")
    print("  dir2_wbc_trajectories_15_26w.png — Direction 2 trajectory plot")
    print("  dir2_group_comparison_table.csv  — Direction 2 stats table")
    print("  metrics_summary.json             — Combined metrics (both directions)")


import math  # needed for nrows calc above

if __name__ == "__main__":
    main()


# In[ ]:




