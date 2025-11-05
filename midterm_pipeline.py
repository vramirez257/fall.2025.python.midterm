#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Midterm configurable pipeline:
- Supports CSV/XLSX input
- Targets: TotalAmount | FinalPrice | InventoryAction | HighRating
- Leakage guardrails by target
- Seasonality features (month + Brand/Region effects) for forecasting if OrderDate exists
- Regression (LinearRegression + tuned RandomForestRegressor)
- Classification (RandomForestClassifier) with optional probability calibration
- Exports metrics, plots, scored rows, feature importances, top combos, what-if sims
- Saves all artifacts into ./outputs/
"""

import os, sys, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless (no GUI)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.metrics import get_scorer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# robust RMSE (sklearn versions differ)
def rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def get_reg_scoring():
    try:
        get_scorer("neg_root_mean_squared_error")
        return "neg_root_mean_squared_error"
    except Exception:
        return "neg_mean_squared_error"

# ensure folder
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# safe savefig
def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------
def read_data(path: Path, excel: bool, na_values: List[str], row_limit: int|None) -> pd.DataFrame:
    if excel:
        df = pd.read_excel(path, sheet_name=0, na_values=na_values)
    else:
        df = pd.read_csv(path, na_values=na_values)
    if row_limit:
        df = df.sample(row_limit, random_state=RANDOM_STATE)
    return df

# ---------------------------------------------------------------------
# Cleaning & Leakage guardrails
# ---------------------------------------------------------------------
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    # strip strings
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    # drop totally empty cols
    df = df.dropna(axis=1, how="all")
    # never use identifiers
    for idcol in ["TransactionID"]:
        if idcol in df.columns:
            df = df.drop(columns=[idcol])
    print(f"Removed {before - len(df)} duplicate rows.")
    return df

def target_aware_drop(X: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Drop columns that algebraically reveal the chosen target."""
    X = X.copy()
    leakers_finalprice = {"Price", "DiscountPct", "TotalAmount"}     # FinalPrice = Price*(1-DiscountPct/100)
    leakers_totalamount = {"Quantity", "FinalPrice", "Price", "DiscountPct"}  # TotalAmount = FinalPrice*Quantity
    if target_name == "FinalPrice":
        X = X.drop(columns=[c for c in leakers_finalprice if c in X.columns], errors="ignore")
    elif target_name == "TotalAmount":
        X = X.drop(columns=[c for c in leakers_totalamount if c in X.columns], errors="ignore")
    return X

def add_seasonality_features(X: pd.DataFrame, df_orig: pd.DataFrame) -> pd.DataFrame:
    """Add safe seasonality features if OrderDate exists: month (1–12) and optional Brand/Region interactions (via OHE later)."""
    X = X.copy()
    if "OrderDate" in df_orig.columns:
        dt = pd.to_datetime(df_orig["OrderDate"], errors="coerce")
        X["Month"] = dt.dt.month
    return X

# ---------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------
def make_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    # handle sparse_output for newer sklearn
    try:
        cat_transformer = Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
    except TypeError:
        cat_transformer = Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre, num_cols, cat_cols

# ---------------------------------------------------------------------
# EDA (saved only)
# ---------------------------------------------------------------------
def eda_plots(df: pd.DataFrame, target_name: str, outdir: Path):
    # histogram for TotalAmount or FinalPrice if present
    for col in [target_name, "TotalAmount", "FinalPrice"]:
        if col in df.columns:
            plt.figure(figsize=(8,5))
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            plt.hist(v, bins=30, edgecolor="black")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col); plt.ylabel("Count")
            savefig(outdir / f"eda_{col.lower()}_hist.png")
            break

    # correlation heatmap for numeric cols (excluding obvious identifiers)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:
        corr = df[num_cols].corr(numeric_only=True)
        plt.figure(figsize=(8,6))
        plt.imshow(corr, aspect="auto", interpolation="nearest", cmap="coolwarm")
        plt.colorbar()
        plt.title("Correlation Heatmap (numeric)")
        plt.xticks(range(len(num_cols)), num_cols, rotation=90, fontsize=8)
        plt.yticks(range(len(num_cols)), num_cols, fontsize=8)
        savefig(outdir / "eda_corr_heatmap.png")

    # bar charts (Brand, Region) if present
    if "Brand" in df.columns:
        plt.figure(figsize=(8,4))
        vc = df["Brand"].value_counts()
        ax = vc.plot(kind="bar")
        plt.title("Sales Count by Brand"); plt.xlabel("Brand"); plt.ylabel("Count")
        savefig(outdir / "eda_bar_brand.png")
    if "Region" in df.columns:
        plt.figure(figsize=(8,4))
        vc = df["Region"].value_counts()
        ax = vc.plot(kind="bar", color="tab:blue")
        plt.title("Sales Count by Region"); plt.xlabel("Region"); plt.ylabel("Count")
        savefig(outdir / "eda_bar_region.png")

# ---------------------------------------------------------------------
# KPI table
# ---------------------------------------------------------------------
def kpi_region_brand(df: pd.DataFrame, outdir: Path):
    cols = [c for c in ["Region", "Brand", "TotalAmount", "FinalPrice", "Quantity"] if c in df.columns]
    if ("Region" in cols or "Brand" in cols):
        group_cols = [c for c in ["Region", "Brand"] if c in df.columns]
        metrics = {}
        if "TotalAmount" in df.columns:
            metrics["TotalAmount"] = ["sum", "mean"]
        if "FinalPrice" in df.columns:
            metrics["FinalPrice"] = ["mean"]
        if "Quantity" in df.columns:
            metrics["Quantity"] = ["sum", "mean"]
        if not metrics:
            return
        kpi = df.groupby(group_cols).agg(metrics)
        kpi.columns = ['_'.join(col).strip() for col in kpi.columns.values]
        kpi = kpi.reset_index()
        kpi.to_csv(outdir / "kpi_region_brand.csv", index=False)

# ---------------------------------------------------------------------
# Regression (TotalAmount / FinalPrice)
# ---------------------------------------------------------------------
def run_regression(df: pd.DataFrame, target_name: str, add_season: bool, outdir: Path):
    assert target_name in df.columns, f"Target '{target_name}' not found."
    y = pd.to_numeric(df[target_name], errors="coerce")
    X = df.drop(columns=[target_name])

    # drop raw dates from features (we'll add Month safely if requested)
    if "OrderDate" in X.columns:
        X = X.drop(columns=["OrderDate"])

    # leakage guardrails for this target
    X = target_aware_drop(X, target_name)
    if add_season:
        X = add_seasonality_features(X, df)

    # split
    stratify_arg = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_arg
    )

    # preprocess
    pre, _, _ = make_preprocessor(X_train)

    results = []

    # Linear Regression
    reg_lr = Pipeline([("pre", pre), ("lr", LinearRegression())])
    reg_lr.fit(X_train, y_train)
    y_pred_lr = reg_lr.predict(X_test)
    results.append({
        "model": "LinearRegression",
        "rmse": rmse(y_test, y_pred_lr),
        "mae": mean_absolute_error(y_test, y_pred_lr),
        "r2": r2_score(y_test, y_pred_lr)
    })

    # RandomForest with light tuning
    reg_rf = Pipeline([("pre", pre), ("rf", RandomForestRegressor(random_state=RANDOM_STATE))])
    param_grid = {
        "rf__n_estimators": [200, 400],
        "rf__max_depth": [None, 10, 20],
        "rf__min_samples_split": [2, 5]
    }
    scoring = get_reg_scoring()
    grid = GridSearchCV(reg_rf, param_grid, cv=3, n_jobs=-1, scoring=scoring)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred_rf = best.predict(X_test)
    results.append({
        "model": "RandomForestRegressor (tuned)",
        "rmse": rmse(y_test, y_pred_rf),
        "mae": mean_absolute_error(y_test, y_pred_rf),
        "r2": r2_score(y_test, y_pred_rf),
        "best_params": grid.best_params_
    })

    res_df = pd.DataFrame(results)
    res_df.to_csv(outdir / f"model_results_{target_name}.csv", index=False)
    print("\n=== Model Results ===")
    print(res_df)

    # plots: residuals & predicted vs actual for RF
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, y_pred_rf, s=10, alpha=0.6)
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(f"Predicted vs Actual — {target_name}")
    savefig(outdir / f"reg_pred_vs_actual_{target_name}.png")

    plt.figure(figsize=(7,5))
    residuals = y_test - y_pred_rf
    plt.hist(residuals, bins=30, edgecolor="black")
    plt.title(f"Residuals — {target_name} (RF)")
    plt.xlabel("Actual - Predicted"); plt.ylabel("Count")
    savefig(outdir / f"reg_residuals_{target_name}.png")

# ---------------------------------------------------------------------
# Time-series style monthly demand forecast (Units), no raw date features
# ---------------------------------------------------------------------
def run_forecast_units(df: pd.DataFrame, outdir: Path):
    if "OrderDate" not in df.columns:
        print("OrderDate not found; skipping demand forecast.")
        return
    df_ts = df.copy()
    df_ts["OrderDate"] = pd.to_datetime(df_ts["OrderDate"], errors="coerce")
    if "Quantity" not in df_ts.columns:
        df_ts["Quantity"] = 1
    df_ts["Quantity"] = pd.to_numeric(df_ts["Quantity"], errors="coerce").fillna(0).astype(float)
    has_rating = "Rating" in df_ts.columns
    if has_rating:
        df_ts["Rating"] = pd.to_numeric(df_ts["Rating"], errors="coerce")

    df_ts["YearMonth"] = df_ts["OrderDate"].dt.to_period("M").dt.to_timestamp()
    group_cols = [c for c in ["Region", "Brand", "Category"] if c in df_ts.columns]
    if not group_cols:
        group_cols = ["Brand"] if "Brand" in df_ts.columns else []

    agg = {"Quantity": "sum"}
    if has_rating: agg["Rating"] = "mean"
    g = df_ts.groupby(group_cols + ["YearMonth"], as_index=False).agg(agg)
    g = g.sort_values(group_cols + ["YearMonth"]).reset_index(drop=True)
    g = g.rename(columns={"Quantity": "Units"})

    def add_lags(df_group, target="Units", lags=(1,2,3), rolls=(2,3)):
        df_group = df_group.copy()
        for k in lags:
            df_group[f"{target}_lag{k}"] = df_group[target].shift(k)
        for w in rolls:
            df_group[f"{target}_roll{w}"] = df_group[target].shift(1).rolling(w).mean()
        if has_rating:
            df_group["Rating_lag1"] = df_group["Rating"].shift(1)
            df_group["Rating_roll3"] = df_group["Rating"].shift(1).rolling(3).mean()
        return df_group

    gf = g.groupby(group_cols, group_keys=False).apply(add_lags).dropna().reset_index(drop=True)

    if gf.empty:
        print("Insufficient data for time-based forecast after lagging.")
        return

    maxm = gf["YearMonth"].max()
    cutoff = maxm - pd.DateOffset(months=6)
    train_mask = gf["YearMonth"] <= cutoff
    gtr, gte = gf[train_mask], gf[~train_mask]
    if gtr.empty or gte.empty:
        # fallback 80/20 on unique months
        uniq = np.sort(gf["YearMonth"].unique())
        if len(uniq) > 2:
            cutidx = int(len(uniq)*0.8)
            cutoff = uniq[cutidx]
            train_mask = gf["YearMonth"] <= cutoff
            gtr, gte = gf[train_mask], gf[~train_mask]
        else:
            print("Too few months for forecast split.")
            return

    feature_cols = [c for c in gf.columns if c not in (group_cols + ["YearMonth", "Units", "Rating"])]
    X_tr, y_tr = gtr[feature_cols], gtr["Units"]
    X_te, y_te = gte[feature_cols], gte["Units"]

    reg = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE)
    reg.fit(X_tr, y_tr)
    yhat = reg.predict(X_te)
    print(f"\nDemand Forecast — RMSE: {rmse(y_te, yhat)} MAE: {mean_absolute_error(y_te, yhat)} R2: {r2_score(y_te, yhat)}")

    forecast_out = gte[group_cols + ["YearMonth"]].copy()
    forecast_out["ActualUnits"] = y_te.values
    forecast_out["PredUnits"] = yhat
    forecast_out = forecast_out.sort_values(group_cols + ["YearMonth"])
    forecast_out.to_csv(outdir / "monthly_demand_forecasts.csv", index=False)

    # one example line chart for first group
    if not forecast_out.empty and group_cols:
        first = tuple(forecast_out[group_cols].iloc[0])
        mask = np.logical_and.reduce([forecast_out[c].eq(v) for c, v in zip(group_cols, first)])
        sub = forecast_out[mask]
        plt.figure(figsize=(10,5))
        plt.plot(sub["YearMonth"], sub["ActualUnits"], marker="o", label="Actual")
        plt.plot(sub["YearMonth"], sub["PredUnits"], marker="s", label="Forecast")
        plt.xticks(rotation=45)
        plt.title(f"Units Forecast — {dict(zip(group_cols, first))}")
        plt.xlabel("Month"); plt.ylabel("Units")
        plt.grid(True, linestyle="--", alpha=0.5); plt.legend()
        savefig(outdir / "forecast_example_line.png")

# ---------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------
def fit_classification_pipeline(X_tr, y_tr, pre, calibrate: bool):
    base = Pipeline([
        ("prep", pre),
        ("rf", RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=2,
            class_weight="balanced", random_state=RANDOM_STATE
        ))
    ])
    base.fit(X_tr, y_tr)
    if calibrate:
        # Calibrate the whole pipeline
        cal = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        cal.fit(X_tr, y_tr)
        return cal
    return base

def evaluate_classifier(clf, X_te, y_te, threshold: float) -> Dict[str, float]:
    proba = clf.predict_proba(X_te)[:, 1]
    yhat = (proba >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_te, yhat),
        "precision": precision_score(y_te, yhat, zero_division=0),
        "recall": recall_score(y_te, yhat, zero_division=0),
        "f1": f1_score(y_te, yhat, zero_division=0)
    }
    try:
        out["auc"] = roc_auc_score(y_te, proba)
    except Exception:
        out["auc"] = float("nan")
    return out, yhat, proba

def save_confusion(y_true, y_pred, title: str, outpath: Path):
    plt.figure(figsize=(5,5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(values_format='d')
    plt.title(title)
    savefig(outpath)

# ---------------------------------------------------------------------
# InventoryAction (Rating>=4 => 1), no Rating in features
# ---------------------------------------------------------------------
def run_inventory(df: pd.DataFrame, threshold: float, calibrate: bool, outdir: Path):
    if "Rating" not in df.columns:
        print("Rating not found; skipping InventoryAction.")
        return
    inv = df.copy()
    inv["Rating"] = pd.to_numeric(inv["Rating"], errors="coerce")
    inv = inv.dropna(subset=["Rating"])
    inv = inv[inv["Rating"].isin([1,2,3,4,5])]

    inv["InventoryAction"] = (inv["Rating"] >= 4).astype(int)
    y = inv["InventoryAction"].astype(int)

    drop_cols = ["InventoryAction", "Rating"]
    X = inv.drop(columns=[c for c in drop_cols if c in inv.columns])

    # drop raw date
    if "OrderDate" in X.columns:
        X = X.drop(columns=["OrderDate"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pre, num_cols, cat_cols = make_preprocessor(X_tr)
    clf = fit_classification_pipeline(X_tr, y_tr, pre, calibrate)

    metrics, yhat, proba = evaluate_classifier(clf, X_te, y_te, threshold)
    print(f"\nInventoryAction — Acc: {metrics['accuracy']:.3f} Prec: {metrics['precision']:.3f} Rec: {metrics['recall']:.3f} F1: {metrics['f1']:.3f} AUC: {metrics['auc']:.3f}")

    # confusion matrix
    save_confusion(y_te, yhat, "InventoryAction Confusion Matrix", outdir / "cm_InventoryAction.png")

    # feature importances (best-effort for RF)
    try:
        # get OHE names
        ohe = clf.base_estimator.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"] \
              if isinstance(clf, CalibratedClassifierCV) else \
              clf.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
        if hasattr(ohe, "get_feature_names_out"):
            cat_ohe_names = list(ohe.get_feature_names_out(cat_cols))
        else:
            cat_ohe_names = list(cat_cols)
        feat_names = list(num_cols) + cat_ohe_names

        rf = clf.base_estimator.named_steps["rf"] if isinstance(clf, CalibratedClassifierCV) else clf.named_steps["rf"]
        importances = getattr(rf, "feature_importances_", None)
        if importances is not None and len(importances) == len(feat_names):
            imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
            imp_df.to_csv(outdir / "feature_importance_InventoryAction.csv", index=False)
    except Exception:
        pass

    # scored rows
    scored = X_te.copy()
    scored["y_true"] = y_te.values
    scored["proba"] = proba
    scored["pred"] = yhat
    scored.to_csv(outdir / "scored_InventoryAction.csv", index=False)

    # top combos Brand×CPU×RAM by avg proba
    keys = [k for k in ["Brand", "CPU", "RAM"] if k in X_te.columns]
    if keys:
        tmp = X_te.copy()
        tmp["proba"] = proba
        top = (tmp.groupby(keys)["proba"].mean().reset_index()
               .sort_values("proba", ascending=False).head(25))
        top.to_csv(outdir / "top_combos_InventoryAction.csv", index=False)

# ---------------------------------------------------------------------
# HighRating (Rating>=4 => 1), no Rating in features + what-if sims
# ---------------------------------------------------------------------
def run_highrating(df: pd.DataFrame, threshold: float, calibrate: bool, outdir: Path):
    if "Rating" not in df.columns:
        print("Rating not found; skipping HighRating.")
        return
    sim = df.copy()
    sim["Rating"] = pd.to_numeric(sim["Rating"], errors="coerce")
    sim = sim.dropna(subset=["Rating"])
    y = (sim["Rating"] >= 4).astype(int)

    X = sim.drop(columns=["Rating"])
    if "OrderDate" in X.columns:
        X = X.drop(columns=["OrderDate"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pre, num_cols, cat_cols = make_preprocessor(X_tr)
    clf = fit_classification_pipeline(X_tr, y_tr, pre, calibrate)

    metrics, yhat, proba = evaluate_classifier(clf, X_te, y_te, threshold)
    print(f"\nHighRating — Acc: {metrics['accuracy']:.3f} Prec: {metrics['precision']:.3f} Rec: {metrics['recall']:.3f} F1: {metrics['f1']:.3f} AUC: {metrics['auc']:.3f}")

    save_confusion(y_te, yhat, "HighRating Confusion Matrix", outdir / "cm_HighRating.png")

    # feature importances
    try:
        ohe = clf.base_estimator.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"] \
              if isinstance(clf, CalibratedClassifierCV) else \
              clf.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
        if hasattr(ohe, "get_feature_names_out"):
            cat_ohe_names = list(ohe.get_feature_names_out(cat_cols))
        else:
            cat_ohe_names = list(cat_cols)
        feat_names = list(num_cols) + cat_ohe_names

        rf = clf.base_estimator.named_steps["rf"] if isinstance(clf, CalibratedClassifierCV) else clf.named_steps["rf"]
        importances = getattr(rf, "feature_importances_", None)
        if importances is not None and len(importances) == len(feat_names):
            imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
            imp_df.to_csv(outdir / "feature_importance_HighRating.csv", index=False)
    except Exception:
        pass

    # scored rows
    scored = X_te.copy()
    scored["y_true"] = y_te.values
    scored["proba"] = proba
    scored["pred"] = yhat
    scored.to_csv(outdir / "scored_HighRating.csv", index=False)

    # top combos Brand×CPU×RAM by avg proba
    keys = [k for k in ["Brand", "CPU", "RAM"] if k in X_te.columns]
    if keys:
        tmp = X_te.copy()
        tmp["proba"] = proba
        top = (tmp.groupby(keys)["proba"].mean().reset_index()
               .sort_values("proba", ascending=False).head(25))
        top.to_csv(outdir / "top_combos_HighRating.csv", index=False)

    # What-if simulation: tweak a single held-out row
    examples = []
    if not X_te.empty:
        row = X_te.iloc[0]
        candidates = []
        if "RAM" in X_te.columns:
            for val in [8, 16, 32]:
                candidates.append({"RAM": val})
        if "Storage" in X_te.columns:
            try:
                base_storage = float(row.get("Storage", 256))
                candidates.append({"Storage": base_storage * 2})
            except Exception:
                candidates.append({"Storage": "512GB SSD"})
        if "CPU" in X_te.columns:
            for v in ["Core i7", "Ryzen 7"]:
                candidates.append({"CPU": v})
        if "GPU" in X_te.columns:
            for v in ["NVIDIA RTX 4060", "NVIDIA RTX 3050"]:
                candidates.append({"GPU": v})
        if "OS" in X_te.columns:
            candidates.append({"OS": "Windows 11 Pro"})

        base_p = float(clf.predict_proba(pd.DataFrame([row]))[:, 1][0])
        for ch in candidates[:10]:
            new_row = row.copy()
            for k, v in ch.items():
                if k in new_row.index:
                    new_row[k] = v
            new_p = float(clf.predict_proba(pd.DataFrame([new_row]))[:, 1][0])
            examples.append({"change": ch, "delta_high_rating_prob": new_p - base_p})

    if examples:
        pd.DataFrame(examples).to_csv(outdir / "what_if_simulations.csv", index=False)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Midterm configurable pipeline")
    ap.add_argument("--data", required=True, help="Path to CSV/XLSX")
    ap.add_argument("--excel", action="store_true", help="Set if the data file is Excel")
    ap.add_argument("--target", default="TotalAmount",
                    help="Regression target or classification alias (TotalAmount | FinalPrice | HighRating | InventoryAction)")
    ap.add_argument("--threshold", type=float, default=0.50, help="Decision threshold for classifiers")
    ap.add_argument("--row-limit", type=int, default=None, help="Optional row limit (sample)")
    ap.add_argument("--add-seasonality", action="store_true",
                    help="Add Month-of-Year (safe) to features where appropriate")
    ap.add_argument("--na", nargs="*", default=["NA","N/A","","?","null","None"], help="Extra NA markers")
    ap.add_argument("--calibrate", action="store_true",
                    help="Calibrate classifier probabilities (sigmoid; better thresholding)")

    args = ap.parse_args()

    # Resolve paths
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path(os.getcwd()).resolve()
    data_path = (here / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    print(f"Resolved data path: {data_path}")

    outdir = here / "outputs"
    ensure_dir(outdir)

    # Load & clean
    df = read_data(data_path, excel=args.excel, na_values=args.na, row_limit=args.row_limit)
    df = basic_clean(df)

    # Save cleaned copy (for reproducibility)
    df.to_csv(outdir / "cleaned_dataset.csv", index=False)

    # EDA + KPIs
    eda_plots(df, args.target, outdir)
    kpi_region_brand(df, outdir)

    # Branch by target
    target = args.target

    if target in ("TotalAmount", "FinalPrice"):
        # regression with leakage guardrails + optional seasonality
        run_regression(df, target, add_season=args.add_seasonality, outdir=outdir)
        # demand forecast (units/month)
        run_forecast_units(df, outdir)

    elif target == "InventoryAction":
        run_inventory(df, threshold=args.threshold, calibrate=args.calibrate, outdir=outdir)

        # (Optional) also echo HighRating with same settings for convenience
        # Comment these 2 lines if you want strict per-target behavior only.
        run_highrating(df, threshold=args.threshold, calibrate=args.calibrate, outdir=outdir)

    elif target == "HighRating":
        run_highrating(df, threshold=args.threshold, calibrate=args.calibrate, outdir=outdir)

    else:
        print(f"Unknown target '{target}'. Nothing to run.")

    # Model card
    model_card = {
        "random_state": RANDOM_STATE,
        "target": target,
        "date_used_as_feature": False,   # we never feed raw dates; only Month when requested
        "leakage_guardrails": True,
        "calibrated": bool(args.calibrate),
        "seasonality": bool(args.add_seasonality)
    }
    (outdir / "model_card.json").write_text(json.dumps(model_card, indent=2))

    # List artifacts
    files = sorted([p.name for p in outdir.iterdir() if p.is_file()])
    print("\nArtifacts saved to:", outdir)
    print("Files:", files)

if __name__ == "__main__":
    sys.exit(main())
