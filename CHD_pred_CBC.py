#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHD prediction from CBC trajectory features (weeks 0–12).
Run:
  python run_chd_metrics.py --base-dir "<PATH_TO_DATA_DIR>"
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Safe import of scikit-learn with clear error ----
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, roc_curve,
        precision_recall_curve, f1_score, confusion_matrix
    )
except Exception as e:
    raise SystemExit(
        "ERROR: scikit-learn is not available in this environment.\n"
        "Install it, then re-run this script.\n\n"
        "  pip install --upgrade scikit-learn\n"
        "  # or\n"
        "  conda install -y scikit-learn\n\n"
        f"Details: {e}"
    )

# ---------------------------
# Helpers
# ---------------------------
def extract_week_number(colname: str) -> Optional[int]:
    """Extract the last integer in a column name as the week number."""
    m = re.search(r"(\d+)(?!.*\d)", colname)
    return int(m.group(1)) if m else None

def week_columns_for_measurement(df: pd.DataFrame, measurement_name: str) -> Dict[int, str]:
    """Return mapping {week: column_name} for a full measurement name."""
    patt = re.escape(measurement_name)
    cols = [c for c in df.columns if re.search(patt, c, flags=re.IGNORECASE)]
    wmap = {}
    for c in cols:
        w = extract_week_number(c)
        if w is not None and 0 <= w <= 26:
            wmap[w] = c
    return dict(sorted(wmap.items()))

def compute_week_reference(df: pd.DataFrame, week_cols: Dict[int, str]) -> pd.DataFrame:
    """Compute reference mean and std per week; fill gaps by interpolation."""
    rows = []
    for w, col in week_cols.items():
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "week": w,
            "mean": float(vals.mean()) if len(vals) else np.nan,
            "std": float(vals.std(ddof=0)) if len(vals) else np.nan
        })
    ref = pd.DataFrame(rows).sort_values("week")
    ref["mean"] = ref["mean"].interpolate(limit_direction="both")
    ref["std"] = ref["std"].replace(0, np.nan).interpolate(limit_direction="both")
    if ref["std"].isna().any():
        ref["std"] = ref["std"].fillna(ref["std"].median())
    return ref

def add_zscores(df: pd.DataFrame, week_cols: Dict[int, str], ref: pd.DataFrame) -> pd.DataFrame:
    """Create <orig_col>__z for each week column using reference stats."""
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
    """Return (mean, slope) for ordered z-scores; slope NaN if <3 points."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    if len(arr) == 0:
        return (np.nan, np.nan)
    meanv = float(np.mean(arr))
    slope = np.nan
    if len(arr) >= 3:
        x = np.arange(len(arr))
        slope = float(np.polyfit(x, arr, 1)[0])
    return (meanv, slope)

def rolling_flag(values: List[float], thresh_abs: float, needed_weeks: int, mode: str = ">= ") -> int:
    """
    Return 1 if condition is met consecutively for needed_weeks; else 0.
    mode: ">= " means |z| >= thresh_abs; "<= " means |z| <= thresh_abs.
    """
    ser = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if mode.strip() == "<=":
        arr = ser.fillna(float("inf")).abs().values  # missing treated as far from center
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
    """Find probability threshold that maximizes F1; return (best_thresh, best_f1)."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = (2 * prec * rec) / np.clip(prec + rec, 1e-12, None)
    # precision_recall_curve returns an extra first point for (prec=pos_rate, rec=1.0) without threshold
    f1s = f1s[1:]  # align with thresholds
    if len(f1s) == 0:
        return 0.5, 0.0
    idx = int(np.argmax(f1s))
    return float(thr[idx]), float(f1s[idx])

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute AUROC/AUPRC for CHD from CBC trajectories (0–12 weeks).")
    parser.add_argument("--base-dir", type=str, required=True, help="Directory containing input CSV files.")
    parser.add_argument("--pre-diag-max-week", type=int, default=12, help="Max week to include (default: 12).")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    # Required files
    pivot_path = base_dir / "babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv"
    labels_path = base_dir / "congenital_circ_labels.csv"
    ga_path = base_dir / "Babies_GA.csv"

    for p in [pivot_path, labels_path]:
        if not p.exists():
            raise SystemExit(f"Required file missing: {p}")

    # Load data
    pivot = pd.read_csv(pivot_path)
    labels = pd.read_csv(labels_path)
    ga = pd.read_csv(ga_path) if ga_path.exists() else None

    PERSON_ID_COL = "person_id"
    if PERSON_ID_COL not in pivot.columns or PERSON_ID_COL not in labels.columns:
        raise SystemExit(f"'{PERSON_ID_COL}' must exist in pivot and labels files.")

    # Detect CHD label column (0/1)
    def find_chd_label_column(labels_df: pd.DataFrame) -> str:
        candidates = [c for c in labels_df.columns if c != PERSON_ID_COL]
        priority = [c for c in candidates if any(k in c.lower() for k in ["chd","cong","circ","card","heart","anomaly","defect"])]
        ordered = priority + [c for c in candidates if c not in priority]
        for col in ordered:
            vals = pd.to_numeric(labels_df[col], errors="coerce").dropna().unique()
            if set(vals).issubset({0,1}):
                return col
        raise SystemExit("Could not auto-detect CHD label column with 0/1 values; please add e.g. 'CHD'.")

    label_col = find_chd_label_column(labels)
    labels = labels[[PERSON_ID_COL, label_col]].rename(columns={label_col: "CHD"})
    labels["CHD"] = pd.to_numeric(labels["CHD"], errors="coerce").fillna(0).astype(int)

    # Optional GA
    if ga is not None and PERSON_ID_COL in ga.columns:
        ga_col = None
        for c in ga.columns:
            if c == PERSON_ID_COL: 
                continue
            cl = c.lower()
            if "gest" in cl or cl == "ga":
                ga_col = c
                break
        if ga_col is None:
            non_id = [c for c in ga.columns if c != PERSON_ID_COL]
            if len(non_id) == 1:
                ga_col = non_id[0]
        if ga_col:
            ga = ga[[PERSON_ID_COL, ga_col]].rename(columns={ga_col: "gestational_age"})
        else:
            ga = None

    # Merge base
    df = pivot.merge(labels, on=PERSON_ID_COL, how="inner")
    if ga is not None:
        df = df.merge(ga, on=PERSON_ID_COL, how="left")
    print(f"[OK] Base merged: {df.shape}")

    # Measurement names
    MEASUREMENT_NAMES = [
        "Neutrophils/100 leukocytes in Blood",
        "Lymphocytes/100 leukocytes in Blood",
        "Monocytes/100 leukocytes in Blood",
        "Eosinophils/100 leukocytes in Blood",
        "Basophils/100 leukocytes in Blood"
    ]

    PRE_DIAG_MAX_WEEK = int(args.pre_diag_max_week)

    # Feature construction
    feature_frames = []
    for meas in MEASUREMENT_NAMES:
        wcols = week_columns_for_measurement(df, meas)
        if not wcols:
            print(f"[WARN] No week columns detected for: {meas}")
            continue

        ref = compute_week_reference(df, wcols)
        df = add_zscores(df, wcols, ref)

        zcols = {w: f"{col}__z" for w, col in wcols.items() if w <= PRE_DIAG_MAX_WEEK and f"{col}__z" in df.columns}
        if not zcols:
            print(f"[WARN] No z-score columns ≤ week {PRE_DIAG_MAX_WEEK} for: {meas}")
            continue

        ordered_weeks = sorted(zcols.keys())
        zlist = [zcols[w] for w in ordered_weeks]
        tag = re.sub(r"[^A-Za-z]+", "_", meas).strip("_").lower()

        means, slopes, devs, convs, rerise = [], [], [], [], []
        for _, row in df.iterrows():
            vals = [row.get(c, np.nan) for c in zlist]
            meanv, slopev = summarize_series(vals)
            means.append(meanv)
            slopes.append(slopev)
            devs.append(rolling_flag(vals, 1.5, 2, mode=">= "))
            convs.append(rolling_flag(vals, 1.0, 3, mode="<= "))
            # neutrophil re-rise alert: slope (12–15) ≤ 0
            if "neutrophil" in meas.lower():
                sub = []
                for w in range(12, 16):
                    colw = zcols.get(w, None)
                    if colw is not None and pd.notna(row.get(colw, np.nan)):
                        sub.append(float(row[colw]))
                if len(sub) >= 3:
                    x = np.arange(len(sub))
                    s = float(np.polyfit(x, sub, 1)[0])
                    rerise.append(1 if s <= 0 else 0)
                else:
                    rerise.append(0)
            else:
                rerise.append(0)

        feat = pd.DataFrame({
            PERSON_ID_COL: df[PERSON_ID_COL].values,
            f"{tag}_mean_0_{PRE_DIAG_MAX_WEEK}w": means,
            f"{tag}_slope_0_{PRE_DIAG_MAX_WEEK}w": slopes,
            f"{tag}_deviation_flag": devs,
            f"{tag}_convergence_flag": convs,
            f"{tag}_re_rise_alert": rerise,
        })
        feature_frames.append(feat)

    if not feature_frames:
        raise SystemExit("No features constructed — check measurement names or pivot column patterns.")

    features = feature_frames[0]
    for k in feature_frames[1:]:
        features = features.merge(k, on=PERSON_ID_COL, how="outer")

    features = features.merge(df[[PERSON_ID_COL, "CHD"]], on=PERSON_ID_COL, how="inner")
    if "gestational_age" in df.columns:
        features = features.merge(df[[PERSON_ID_COL, "gestational_age"]], on=PERSON_ID_COL, how="left")

    features.to_csv("chd_features_0_12w.csv", index=False)
    print(f"[OK] Saved: chd_features_0_12w.csv  shape={features.shape}")

    # Modeling
    drop_cols = {PERSON_ID_COL, "CHD"}
    X_cols = [c for c in features.columns if c not in drop_cols and features[c].dtype != "O"]
    X = features[X_cols].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = features["CHD"].astype(int)

    valid_rows = ~X.isna().all(axis=1)
    X, y = X.loc[valid_rows], y.loc[valid_rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=5000, solver="lbfgs")
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    prevalence = float(y_test.mean())

    print("\n=== CHD prediction from CBC (pre-diagnosis 0–12w) ===")
    print(f"Test n = {len(y_test)} | CHD prevalence = {prevalence:.3f}")
    print(f"AUROC  = {auroc:.3f}")
    print(f"AUPRC  = {auprc:.3f} (baseline = {prevalence:.3f})")

    # Curves
    fpr, tpr, _ = roc_curve(y_test, probs)
    prec, rec, thr = precision_recall_curve(y_test, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUROC={auroc:.2f})")
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC – CHD prediction (CBC weeks 0–12)")
    plt.legend(); plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"PR (AUPRC={auprc:.2f})")
    plt.hlines(prevalence, 0, 1, linestyles="--", label="Baseline prevalence")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall – CHD prediction (CBC weeks 0–12)")
    plt.legend(); plt.tight_layout()
    plt.savefig("pr_curve.png", dpi=160)
    plt.close()

    # Best-F1 operating point + confusion matrix
    best_thr, best_f1 = best_f1_threshold(y_test.values, probs)
    y_pred = (probs >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    summary = {
        "test_size": int(len(y_test)),
        "prevalence": prevalence,
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "AUPRC_baseline": prevalence,
        "best_F1_threshold": best_thr,
        "best_F1": best_f1,
        "confusion_matrix_at_bestF1": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }
    with open("metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: roc_curve.png, pr_curve.png, metrics_summary.json")
    print(f"Best-F1 threshold = {best_thr:.3f} | F1 = {best_f1:.3f}")
    print(f"Confusion matrix (tn, fp, fn, tp) = ({tn}, {fp}, {fn}, {tp})")

if __name__ == "__main__":
    main()

