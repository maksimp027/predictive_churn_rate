"""
Train & Evaluation Pipeline for SaaS Churn Prediction.
Trains 3 models (Baseline LogReg, CatBoost, XGBoost), compares metrics,
saves Champion Model and test split for subsequent SHAP/ROI analysis.
"""

import os
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# --- Configuration ---

matplotlib.use("Agg")  # No GUI — save plots to disk
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = "model_ready_data.csv"
RESULTS_DIR = "model_results"
TEST_SPLIT_DIR = os.path.join("data", "test_split")
RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEST_SPLIT_DIR, exist_ok=True)


# =============================================================================
# STEP 1: DATA LOADING AND STRATIFIED SPLIT
# =============================================================================
print("=" * 70)
print("STEP 1: Data loading and stratified Train/Test split")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"[OK] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# --- 1.1 Remove zero-variance and multicollinear features ---
DROP_FEATURES = []

# has_open_ticket — zero variance (all values = 0)
if "has_open_ticket" in df.columns and df["has_open_ticket"].nunique() <= 1:
    DROP_FEATURES.append("has_open_ticket")

# arr_amount — exact copy of mrr_amount × 12 (perfect correlation)
if "arr_amount" in df.columns and "mrr_amount" in df.columns:
    corr = df["arr_amount"].corr(df["mrr_amount"])
    if abs(corr) > 0.999:
        DROP_FEATURES.append("arr_amount")

if DROP_FEATURES:
    df = df.drop(columns=DROP_FEATURES)
    print(f"[INFO] Removed problematic features: {DROP_FEATURES}")
    print(f"       Shape after cleanup: {df.shape}")

# --- 1.2 Separate target variable ---
TARGET = "churn_flag"
y = df[TARGET].copy()
X = df.drop(columns=[TARGET])

print(f"\n[INFO] Target variable: '{TARGET}'")
print(f"       Features (X): {X.shape[1]} columns")
print(f"       y distribution: {y.value_counts().to_dict()}")
print(f"       Churn rate: {y.mean():.2%}")

# --- 1.3 Stratified split 80/20 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"\n[OK] Train/Test split ({1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}):")
print(f"     Train: {X_train.shape[0]} rows | churn={y_train.mean():.2%}")
print(f"     Test:  {X_test.shape[0]} rows | churn={y_test.mean():.2%}")


# =============================================================================
# STEP 2: BASELINE — LOGISTIC REGRESSION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Baseline — LogisticRegression (class_weight='balanced')")
print("=" * 70)

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=RANDOM_STATE,
)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

print("[OK] LogisticRegression trained.")
print(f"     ROC-AUC = {roc_auc_score(y_test, y_prob_lr):.4f}")


# =============================================================================
# STEP 3: CATBOOST
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: CatBoostClassifier (auto_class_weights='Balanced')")
print("=" * 70)

cb_model = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    auto_class_weights="Balanced",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=100,  # Log every 100 iterations
)
cb_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

y_pred_cb = cb_model.predict(X_test).astype(int)
y_prob_cb = cb_model.predict_proba(X_test)[:, 1]

best_iter = cb_model.get_best_iteration()
print(f"\n[OK] CatBoost trained. Best iteration: {best_iter}")
print(f"     ROC-AUC = {roc_auc_score(y_test, y_prob_cb):.4f}")


# =============================================================================
# STEP 4: XGBOOST
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: XGBClassifier (scale_pos_weight)")
print("=" * 70)

# Calculate scale_pos_weight: n_negative / n_positive
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"[INFO] scale_pos_weight = {n_neg}/{n_pos} = {scale_pos_weight:.2f}")

xgb_model = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=RANDOM_STATE,
    verbosity=1,
    early_stopping_rounds=50,
)
xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print(f"\n[OK] XGBoost trained.")
print(f"     ROC-AUC = {roc_auc_score(y_test, y_prob_xgb):.4f}")


# =============================================================================
# STEP 5: COMPARATIVE METRICS TABLE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Model comparison on test set")
print("=" * 70)

models = {
    "LogisticRegression": {"y_pred": y_pred_lr, "y_prob": y_prob_lr, "obj": lr_model},
    "CatBoost": {"y_pred": y_pred_cb, "y_prob": y_prob_cb, "obj": cb_model},
    "XGBoost": {"y_pred": y_pred_xgb, "y_prob": y_prob_xgb, "obj": xgb_model},
}

metrics_rows = []
for name, data in models.items():
    metrics_rows.append(
        {
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, data["y_prob"]),
            "F1": f1_score(y_test, data["y_pred"]),
            "Precision": precision_score(y_test, data["y_pred"]),
            "Recall": recall_score(y_test, data["y_pred"]),
        }
    )

metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
metrics_df = metrics_df.sort_values("ROC-AUC", ascending=False)

print("\n" + metrics_df.to_string(float_format="{:.4f}".format))

# --- Determine Champion Model by ROC-AUC ---
champion_name = metrics_df.index[0]
champion_metrics = metrics_df.iloc[0]
champion_model = models[champion_name]["obj"]
champion_y_pred = models[champion_name]["y_pred"]
champion_y_prob = models[champion_name]["y_prob"]

print(f"\n{'-' * 50}")
print(f">>> Champion Model: {champion_name}")
print(f"   ROC-AUC = {champion_metrics['ROC-AUC']:.4f}")
print(f"   F1      = {champion_metrics['F1']:.4f}")
print(f"   Prec    = {champion_metrics['Precision']:.4f}")
print(f"   Recall  = {champion_metrics['Recall']:.4f}")
print(f"{'-' * 50}")

# --- Classification Report for Champion Model ---
print(f"\nClassification Report ({champion_name}):")
print(classification_report(y_test, champion_y_pred, target_names=["Retained", "Churned"]))


# =============================================================================
# STEP 6: SAVING PLOTS
# =============================================================================
print("=" * 70)
print("STEP 6: Saving plots (Confusion Matrix + ROC curves)")
print("=" * 70)

# --- 6.1 Confusion Matrix for Champion Model ---
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    champion_y_pred,
    display_labels=["Retained (0)", "Churned (1)"],
    cmap="Blues",
    ax=ax,
)
ax.set_title(
    f"Confusion Matrix — {champion_name}\n"
    f"ROC-AUC={champion_metrics['ROC-AUC']:.4f} | F1={champion_metrics['F1']:.4f}",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_champion.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"[OK] Confusion Matrix saved: {cm_path}")

# --- 6.2 ROC curves for all three models ---
fig, ax = plt.subplots(figsize=(8, 7))
colors = {"LogisticRegression": "#7f8c8d", "CatBoost": "#e74c3c", "XGBoost": "#2980b9"}

for name, data in models.items():
    fpr, tpr, _ = roc_curve(y_test, data["y_prob"])
    auc_val = roc_auc_score(y_test, data["y_prob"])
    label = f"{name} (AUC={auc_val:.4f})"
    lw = 2.5 if name == champion_name else 1.5
    ls = "-" if name == champion_name else "--"
    ax.plot(fpr, tpr, label=label, color=colors[name], linewidth=lw, linestyle=ls)

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random (AUC=0.5)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves: Model Comparison", fontsize=14, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(RESULTS_DIR, "roc_curves_comparison.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"[OK] ROC curves saved: {roc_path}")


# =============================================================================
# STEP 7: MODEL AND TEST SPLIT SERIALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: Serializing Champion Model and test split")
print("=" * 70)

# --- 7.1 Save Champion Model ---
model_path = os.path.join(RESULTS_DIR, "champion_churn_model.pkl")

if champion_name == "CatBoost":
    # CatBoost has its own serializer, but pkl is also supported via joblib
    cb_native_path = os.path.join(RESULTS_DIR, "champion_churn_model.cbm")
    champion_model.save_model(cb_native_path)
    print(f"[OK] CatBoost native format: {cb_native_path}")

# Universal pkl for any model
joblib.dump(champion_model, model_path)
print(f"[OK] Champion Model ({champion_name}) saved: {model_path}")

# --- 7.2 Save test split for SHAP / ROI ---
X_test.to_csv(os.path.join(TEST_SPLIT_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(TEST_SPLIT_DIR, "y_test.csv"), index=False)
print(f"[OK] Test split saved: {TEST_SPLIT_DIR}/")
print(f"     X_test: {X_test.shape}")
print(f"     y_test: {y_test.shape}")

# --- 7.3 Save metrics table ---
metrics_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
metrics_df.to_csv(metrics_path)
print(f"[OK] Metrics table saved: {metrics_path}")


# =============================================================================
# FINAL REPORT
# =============================================================================
print("\n" + "=" * 70)
print("PIPELINE COMPLETED")
print("=" * 70)
print(f"""
Artifacts:
  |-- {cm_path}
  |-- {roc_path}
  |-- {metrics_path}
  |-- {model_path}
  |-- {TEST_SPLIT_DIR}/X_test.csv
  +-- {TEST_SPLIT_DIR}/y_test.csv

Champion Model: {champion_name}
  ROC-AUC  = {champion_metrics['ROC-AUC']:.4f}
  F1       = {champion_metrics['F1']:.4f}
  Precision = {champion_metrics['Precision']:.4f}
  Recall   = {champion_metrics['Recall']:.4f}
""")
