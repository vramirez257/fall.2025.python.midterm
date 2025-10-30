# -------------------------------------------------------------------------
# commit: setup-and-imports
# -------------------------------------------------------------------------
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, get_scorer,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Set some global display options
plt.rcParams.update({'figure.figsize': (8, 5)})
pd.set_option('display.max_columns', 100)

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# -------------------------------------------------------------------------
# commit: compatibility-helpers
# -------------------------------------------------------------------------
def rmse_compat(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def get_regression_scoring_name():
    try:
        get_scorer("neg_root_mean_squared_error")
        return "neg_root_mean_squared_error"
    except Exception:
        return "neg_mean_squared_error"


# -------------------------------------------------------------------------
# commit: config-and-paths
# -------------------------------------------------------------------------
CONFIG = {
    "data_path": "laptop_pc_sales_dataset.csv",   # relative path
    "target": "FinalPrice",
    "row_limit": None,
    "na_values": ["NA", "N/A", "", "?", "null", "None"]
}

# Resolve paths whether this runs as a script or in a notebook
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path(os.getcwd()).resolve()

DATA_PATH = (HERE / CONFIG["data_path"]).resolve()


# -------------------------------------------------------------------------
# commit: load-and-inspect
# -------------------------------------------------------------------------
read_kwargs = {}
if CONFIG["na_values"]:
    read_kwargs["na_values"] = CONFIG["na_values"]

df = pd.read_csv(DATA_PATH, **read_kwargs)

if CONFIG["row_limit"]:
    df = df.sample(CONFIG["row_limit"], random_state=RANDOM_STATE)

print("=== Initial Data Inspection ===")
print("Shape:", df.shape)
print(df.head())

print("\n=== df.info ===")
print(df.info())
print("\n=== describe (all) ===")
print(df.describe(include='all').T)


# -------------------------------------------------------------------------
# commit: basic-cleaning
# -------------------------------------------------------------------------
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"\nRemoved {before - after} duplicate rows.")

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# Explicitly drop TransactionID if present (never use as a feature)
if "TransactionID" in df.columns:
    df = df.drop(columns=["TransactionID"])

# Check missingness
missing_ratio = df.isna().mean().sort_values(ascending=False)
print("\nMissing ratio (top 10):")
print(missing_ratio.head(10))

# Optionally drop columns with >60% missing values
THRESH = 0.60
to_drop = missing_ratio[missing_ratio > THRESH].index.tolist()
if to_drop:
    print("Dropping columns due to high missingness:", to_drop)
    df = df.drop(columns=to_drop)

print("Shape after cleaning:", df.shape)


# -------------------------------------------------------------------------
# commit: target-and-split (no date features)
# -------------------------------------------------------------------------
assert CONFIG["target"] is not None, "Set CONFIG['target'] to your target column name."
assert CONFIG["target"] in df.columns, f"Target '{CONFIG['target']}' not found in columns."

y = df[CONFIG["target"]]
X = df.drop(columns=[CONFIG["target"]])

# Do NOT engineer any date-based features; drop OrderDate entirely if present
if "OrderDate" in X.columns:
    X = X.drop(columns=["OrderDate"])

# Defensive: drop all-NA columns
X = X.dropna(axis=1, how='all')

# De-leak features that algebraically reveal FinalPrice
leakers = {"Price", "DiscountPct", "TotalAmount"}  # Price*(1-DiscountPct/100)=FinalPrice; FinalPrice*Quantity=TotalAmount
X = X.drop(columns=[c for c in leakers if c in X.columns])

# Determine task type (FinalPrice => regression)
is_classification = (y.dtype == object) or ((y.dtype.kind in "biu") and (y.nunique() < max(50, int(0.2 * len(y)))))
print("\nTask type:", "Classification" if is_classification else "Regression")
stratify_arg = y if (is_classification and y.nunique() > 1) else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_arg
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("\n=== X_train.describe (all) ===")
print(X_train.describe(include='all').T)


# -------------------------------------------------------------------------
# commit: preprocessing-two-techniques (no date in features)
# -------------------------------------------------------------------------
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print("\nCategorical features:", cat_cols[:10], "..." if len(cat_cols) > 10 else "")
print("Numeric features:", num_cols[:10], "..." if len(num_cols) > 10 else "")

numeric_transformer = Pipeline(steps=[
    ("imputer",  SimpleImputer(strategy="median")),
    ("scaler",   StandardScaler())
])

# Handle sklearn versions: sparse_output (>=1.2) vs sparse (older)
try:
    categorical_transformer = Pipeline(steps=[
        ("imputer",  SimpleImputer(strategy="most_frequent")),
        ("encoder",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
except TypeError:
    categorical_transformer = Pipeline(steps=[
        ("imputer",  SimpleImputer(strategy="most_frequent")),
        ("encoder",  OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)


# -------------------------------------------------------------------------
# commit: light-eda-plots
# -------------------------------------------------------------------------
for col in X_train.select_dtypes(include=[np.number]).columns[:3]:
    plt.figure()
    X_train[col].hist(bins=30)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

if len(num_cols) > 0:
    c = num_cols[0]
    plt.figure()
    plt.boxplot(X_train[c].dropna(), vert=True)
    plt.title(f"Box Plot of {c}")
    plt.ylabel(c)
    plt.tight_layout()
    plt.show()

if len(num_cols) > 1:
    corr = X_train[num_cols].corr()
    plt.figure()
    plt.imshow(corr, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title("Correlation Heatmap (numeric features)")
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# commit: models-and-evaluation (FinalPrice regression)
# -------------------------------------------------------------------------
results = []

# Linear Regression
reg_lr = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])
reg_lr.fit(X_train, y_train)
y_pred_lr = reg_lr.predict(X_test)

results.append({
    "model": "LinearRegression",
    "rmse": rmse_compat(y_test, y_pred_lr),
    "mae": mean_absolute_error(y_test, y_pred_lr),
    "r2": r2_score(y_test, y_pred_lr)
})

# RandomForestRegressor with light tuning
reg_rf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(random_state=RANDOM_STATE))
])

param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

scoring_name = get_regression_scoring_name()
grid = GridSearchCV(reg_rf, param_grid, cv=3, n_jobs=-1, scoring=scoring_name)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)

results.append({
    "model": "RandomForestRegressor (tuned)",
    "rmse": rmse_compat(y_test, y_pred_rf),
    "mae": mean_absolute_error(y_test, y_pred_rf),
    "r2": r2_score(y_test, y_pred_rf),
    "best_params": grid.best_params_
})

print("\n=== Model Results (TransactionID & Date excluded) ===")
print(pd.DataFrame(results))


# =========================================================================
# GOAL 1: Demand Forecasting (date ONLY for ordering/split; NOT a feature)
# =========================================================================
print("\n=== Goal 1: Demand Forecasting (monthly units; date not a feature) ===")

df_ts = df.copy()

# Require an OrderDate to order time, but never feed into the model
if "OrderDate" not in df_ts.columns:
    print("OrderDate not found; skipping Goal 1 (needs date to order months, not as a feature).")
    forecast_out = pd.DataFrame()
else:
    df_ts["OrderDate"] = pd.to_datetime(df_ts["OrderDate"], errors="coerce")

    # Quantity default
    if "Quantity" not in df_ts.columns:
        df_ts["Quantity"] = 1
    df_ts["Quantity"] = pd.to_numeric(df_ts["Quantity"], errors="coerce").fillna(0).astype(float)

    # Ratings if present (used only as *lagged* signals)
    has_rating = "Rating" in df_ts.columns
    if has_rating:
        df_ts["Rating"] = pd.to_numeric(df_ts["Rating"], errors="coerce")

    # Month index (for grouping/splitting only)
    df_ts["YearMonth"] = df_ts["OrderDate"].dt.to_period("M").dt.to_timestamp()

    # Grain for aggregation (no TransactionID, no raw dates)
    group_cols = [c for c in ["Region", "Brand", "Category"] if c in df_ts.columns]
    if not group_cols:
        group_cols = ["Brand"] if "Brand" in df_ts.columns else []

    agg_dict = {"Quantity": "sum"}
    if has_rating:
        agg_dict["Rating"] = "mean"

    g = df_ts.groupby(group_cols + ["YearMonth"], as_index=False).agg(agg_dict)
    g = g.sort_values(group_cols + ["YearMonth"]).reset_index(drop=True)
    g.rename(columns={"Quantity": "Units"}, inplace=True)

    # Build lags/rolls per group (only lagged target/ratings; NO month/year features)
    def add_lags_rolls(df_group, target_col="Units", k_lags=(1,2,3), rolling_windows=(2,3)):
        df_group = df_group.copy()
        for k in k_lags:
            df_group[f"{target_col}_lag{k}"] = df_group[target_col].shift(k)
        for w in rolling_windows:
            df_group[f"{target_col}_roll{w}"] = df_group[target_col].shift(1).rolling(w).mean()
        if has_rating:
            df_group["Rating_lag1"]  = df_group["Rating"].shift(1)
            df_group["Rating_roll3"] = df_group["Rating"].shift(1).rolling(3).mean()
        return df_group

    g_feat = g.groupby(group_cols, group_keys=False).apply(add_lags_rolls)
    g_feat = g_feat.dropna().reset_index(drop=True)

    # Time split by last 6 calendar months (date not used as feature)
    if not g_feat.empty:
        max_month = g_feat["YearMonth"].max()
        cutoff_date = max_month - pd.DateOffset(months=6)
        train_mask = g_feat["YearMonth"] <= cutoff_date
        g_train, g_test = g_feat[train_mask], g_feat[~train_mask]

        # Fallback if short timeline
        if g_test.empty or g_train.empty:
            unique_months = np.sort(g_feat["YearMonth"].unique())
            if len(unique_months) > 2:
                cutoff_idx = int(len(unique_months) * 0.8)
                cutoff_date = unique_months[cutoff_idx]
                train_mask = g_feat["YearMonth"] <= cutoff_date
                g_train, g_test = g_feat[train_mask], g_feat[~train_mask]
            else:
                g_train, g_test = g_feat.copy(), g_feat.iloc[0:0].copy()

        feature_cols = [c for c in g_feat.columns
                        if c not in (group_cols + ["YearMonth", "Units", "Rating"])]

        X_tr = g_train[feature_cols].copy()
        y_tr = g_train["Units"].copy()
        X_te = g_test[feature_cols].copy()
        y_te = g_test["Units"].copy()

        if not X_te.empty:
            ts_reg = RandomForestRegressor(
                n_estimators=400, max_depth=None, min_samples_split=2, random_state=RANDOM_STATE
            )
            ts_reg.fit(X_tr, y_tr)
            y_pred_units = ts_reg.predict(X_te)

            print("Forecast RMSE:", rmse_compat(y_te, y_pred_units))
            print("Forecast MAE:", mean_absolute_error(y_te, y_pred_units))
            print("Forecast R2 :", r2_score(y_te, y_pred_units))

            forecast_out = g_test[group_cols + ["YearMonth"]].copy()
            forecast_out["ActualUnits"] = y_te.values
            forecast_out["PredUnits"]  = y_pred_units
            forecast_out = forecast_out.sort_values(group_cols + ["YearMonth"])
            print("\nSample forecast rows:")
            print(forecast_out.head())

            # Simple visual for one group (x-axis is time index only)
            if not forecast_out.empty:
                grp_example = tuple(forecast_out[group_cols].iloc[0]) if group_cols else None
                if group_cols:
                    mask = np.logical_and.reduce([
                        forecast_out[c].eq(v) for c, v in zip(group_cols, grp_example)
                    ])
                    plt.figure()
                    subset = forecast_out[mask]
                    plt.plot(subset["YearMonth"], subset["ActualUnits"], label="Actual")
                    plt.plot(subset["YearMonth"], subset["PredUnits"],  label="Forecast")
                    plt.title(f"Units Forecast: {dict(zip(group_cols, grp_example))}")
                    plt.xlabel("Month")
                    plt.ylabel("Units")
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
        else:
            forecast_out = pd.DataFrame()
            print("Not enough monthly data after lagging to perform a time-based test split.")
    else:
        forecast_out = pd.DataFrame()
        print("No data available for time-based aggregation after cleaning.")


# =========================================================================
# GOAL 2: Inventory Optimization via Rating Classification (no date features)
# =========================================================================
print("\n=== Goal 2: Inventory Optimization (Rating -> Action; no date features) ===")

has_rating_col = "Rating" in df.columns
if has_rating_col:
    inv = df.copy()
    inv["Rating"] = pd.to_numeric(inv["Rating"], errors="coerce")
    inv = inv.dropna(subset=["Rating"])
    inv = inv[inv["Rating"].isin([1,2,3,4,5])]
    inv = inv[inv["Rating"] != 3]  # optional: remove neutral
    inv["InventoryAction"] = np.where(inv["Rating"] >= 4, 1, 0)

    y_inv = inv["InventoryAction"].astype(int)
    drop_cols = ["InventoryAction", "Rating", CONFIG["target"]]
    drop_cols = [c for c in drop_cols if c in inv.columns]
    X_inv = inv.drop(columns=drop_cols)

    # Ensure no date field sneaks in
    if "OrderDate" in X_inv.columns:
        X_inv = X_inv.drop(columns=["OrderDate"])

    cat_inv = X_inv.select_dtypes(include=['object', 'category']).columns.tolist()
    num_inv = X_inv.select_dtypes(include=[np.number]).columns.tolist()

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    try:
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    except TypeError:
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])

    prep_inv = ColumnTransformer(
        [("num", num_pipe, num_inv), ("cat", cat_pipe, cat_inv)],
        remainder="drop",
        verbose_feature_names_out=False
    )

    clf = Pipeline([
        ("prep", prep_inv),
        ("rf", RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=2, random_state=RANDOM_STATE, class_weight="balanced"
        ))
    ])

    X_inv_train, X_inv_test, y_inv_train, y_inv_test = train_test_split(
        X_inv, y_inv, test_size=0.2, random_state=RANDOM_STATE, stratify=y_inv
    )

    clf.fit(X_inv_train, y_inv_train)
    yhat = clf.predict(X_inv_test)
    yproba = clf.predict_proba(X_inv_test)[:, 1]

    print("Accuracy:", accuracy_score(y_inv_test, yhat))
    print("Precision:", precision_score(y_inv_test, yhat, zero_division=0))
    print("Recall:", recall_score(y_inv_test, yhat, zero_division=0))
    print("F1:", f1_score(y_inv_test, yhat, zero_division=0))
    try:
        print("ROC AUC:", roc_auc_score(y_inv_test, yproba))
    except Exception:
        pass

    plt.figure()
    ConfusionMatrixDisplay(confusion_matrix(y_inv_test, yhat)).plot(values_format='d')
    plt.title("InventoryAction Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Feature importances
    rf_model = clf.named_steps["rf"]
    ohe = clf.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
    if hasattr(ohe, "get_feature_names_out"):
        cat_ohe_names = list(ohe.get_feature_names_out(cat_inv))
    else:
        cat_ohe_names = list(cat_inv)

    feat_names = list(num_inv) + cat_ohe_names
    importances = getattr(rf_model, "feature_importances_", None)
    if importances is not None and len(importances) == len(feat_names):
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
        plt.figure()
        imp_df.head(20).plot.bar(x="feature", y="importance", legend=False)
        plt.title("Top Feature Importances – InventoryAction")
        plt.tight_layout()
        plt.show()

    inv_recs = X_inv_test.copy()
    inv_recs["RecommendIncreaseStock"] = yhat
    inv_recs["Confidence"] = yproba
    print("\nSample inventory recommendations:")
    print(inv_recs.head())
else:
    inv_recs = pd.DataFrame()
    print("Rating column not found; skipping Goal 2.")


# =========================================================================
# GOAL 3: What-If Spec Simulation for High Rating (no date features)
# =========================================================================
print("\n=== Goal 3: What-If Spec Simulation (predict high rating; no date features) ===")

if has_rating_col:
    sim = df.copy()
    sim["Rating"] = pd.to_numeric(sim["Rating"], errors="coerce")
    sim = sim.dropna(subset=["Rating"])
    sim["HighRating"] = (sim["Rating"] >= 4).astype(int)

    drop_cols = ["HighRating", "Rating", CONFIG["target"]]
    drop_cols = [c for c in drop_cols if c in sim.columns]
    X_sim = sim.drop(columns=drop_cols)

    # Ensure no date field sneaks in
    if "OrderDate" in X_sim.columns:
        X_sim = X_sim.drop(columns=["OrderDate"])

    y_sim = sim["HighRating"]

    cat_sim = X_sim.select_dtypes(include=['object', 'category']).columns.tolist()
    num_sim = X_sim.select_dtypes(include=[np.number]).columns.tolist()

    try:
        ohe_sim = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe_sim = OneHotEncoder(handle_unknown="ignore", sparse=False)

    prep_sim = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num_sim),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", ohe_sim)]), cat_sim)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    sim_clf = Pipeline([
        ("prep", prep_sim),
        ("rf", RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=2, random_state=RANDOM_STATE, class_weight="balanced"
        ))
    ])

    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X_sim, y_sim, test_size=0.2, random_state=RANDOM_STATE, stratify=y_sim
    )
    sim_clf.fit(Xs_tr, ys_tr)
    base_auc = roc_auc_score(ys_te, sim_clf.predict_proba(Xs_te)[:, 1])
    print("HighRating model AUC:", round(base_auc, 3))

    # What-If helper (no date handling)
    def simulate_changes(row: pd.Series, changes: dict) -> float:
        base_df = pd.DataFrame([row])
        new_df  = pd.DataFrame([row.copy()])
        for k, v in changes.items():
            if k in new_df.columns:
                new_df.at[new_df.index[0], k] = v
        base_p = float(sim_clf.predict_proba(base_df)[:, 1][0])
        new_p  = float(sim_clf.predict_proba(new_df)[:, 1][0])
        return new_p - base_p

    what_if_examples = []
    if not Xs_te.empty:
        example_row = Xs_te.iloc[0]
        candidate_changes = []
        if "RAM" in Xs_te.columns:
            candidate_changes.append({"RAM": 16})
            candidate_changes.append({"RAM": 32})
        if "Storage" in Xs_te.columns:
            if np.issubdtype(pd.Series(Xs_te["Storage"]).dtype, np.number):
                base_storage = example_row.get("Storage", 256)
                try:
                    base_storage = float(base_storage)
                except Exception:
                    base_storage = 256.0
                candidate_changes.append({"Storage": base_storage * 2})
            else:
                candidate_changes.append({"Storage": "512GB SSD"})
        if "CPU" in Xs_te.columns:
            candidate_changes.append({"CPU": "Core i7"})
            candidate_changes.append({"CPU": "Ryzen 7"})
        if "GPU" in Xs_te.columns:
            candidate_changes.append({"GPU": "NVIDIA RTX 4060"})
        if "OS" in Xs_te.columns:
            candidate_changes.append({"OS": "Windows 11 Pro"})

        for ch in candidate_changes[:8]:
            delta = simulate_changes(example_row, ch)
            what_if_examples.append({"change": ch, "delta_high_rating_prob": delta})

        print("\nWhat-If simulation examples (ΔP[HighRating]):")
        for rec in what_if_examples:
            print(rec)

    what_if_df = pd.DataFrame(what_if_examples)
else:
    print("Rating column not found; skipping Goal 3.")
    what_if_df = pd.DataFrame()


# -------------------------------------------------------------------------
# commit: save-outputs
# -------------------------------------------------------------------------
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False)
pd.DataFrame(results).to_csv(OUTPUT_DIR / "model_results.csv", index=False)

try:
    if 'forecast_out' in locals() and not forecast_out.empty:
        forecast_out.to_csv(OUTPUT_DIR / "monthly_demand_forecasts.csv", index=False)
except Exception as e:
    print("Skipping save of monthly_demand_forecasts.csv:", e)

try:
    if 'inv_recs' in locals() and not inv_recs.empty:
        inv_recs.to_csv(OUTPUT_DIR / "inventory_recommendations.csv", index=False)
except Exception as e:
    print("Skipping save of inventory_recommendations.csv:", e)

try:
    if not what_if_df.empty:
        what_if_df.to_csv(OUTPUT_DIR / "what_if_simulations.csv", index=False)
except Exception as e:
    print("Skipping save of what_if_simulations.csv:", e)

model_card = {
    "random_state": RANDOM_STATE,
    "finalprice_models": results,
    "demand_forecast_features": [c for c in locals().get("feature_cols", [])],  # lags/rolls only
    "inventory_model": "RandomForestClassifier (balanced)" if has_rating_col else None,
    "what_if_model": "RandomForestClassifier (HighRating)" if has_rating_col else None,
    "date_used_as_feature": False,
    "transaction_id_used": False
}
with open(OUTPUT_DIR / "model_card.json", "w") as f:
    json.dump(model_card, f, indent=2)

print("\nSaved files:", [p.name for p in OUTPUT_DIR.iterdir()])
