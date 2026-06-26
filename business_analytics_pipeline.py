"""
Business Analytics Pipeline: XAI (SHAP) + Financial ROI Simulation.
Connects Champion Model technical predictions with real revenue impact.
"""

import os
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# --- Configuration ---

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

RESULTS_DIR = "business_results"
MODEL_PATH = "model_results/champion_churn_model.cbm"
TEST_X_PATH = "data/test_split/X_test.csv"
TEST_Y_PATH = "data/test_split/y_test.csv"
ORIGINAL_DATA_PATH = "data/saas_master_dataset_x6.csv"
MODEL_READY_PATH = "model_ready_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Business parameters for ROI simulation
DISCOUNT_RATE = 0.20       # 20% retention discount
RETENTION_RATE = 0.40      # 40% of clients agree to stay
TOP_RISK_PERCENTILE = 0.20 # Top-20% highest risk

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# STEP 1: LOAD ARTIFACTS
# =============================================================================
print("=" * 70)
print("STEP 1: Loading model, test data, and original dataset")
print("=" * 70)

# --- 1.1 CatBoost model ---
model = CatBoostClassifier()
model.load_model(MODEL_PATH)
print(f"[OK] Model loaded: {MODEL_PATH}")
print(f"     Number of trees: {model.tree_count_}")

# --- 1.2 Test splits (scaled) ---
X_test = pd.read_csv(TEST_X_PATH)
y_test = pd.read_csv(TEST_Y_PATH).squeeze()
print(f"[OK] X_test: {X_test.shape}")
print(f"[OK] y_test: {y_test.shape}, churn={y_test.mean():.2%}")

# --- 1.3 Reproduce test-split indices to access original values ---
# Repeat the exact same split as in train_pipeline.py
df_model = pd.read_csv(MODEL_READY_PATH)

# Drop the same columns as in train_pipeline
DROP_FEATURES = []
if "has_open_ticket" in df_model.columns and df_model["has_open_ticket"].nunique() <= 1:
    DROP_FEATURES.append("has_open_ticket")
if "arr_amount" in df_model.columns and "mrr_amount" in df_model.columns:
    corr = df_model["arr_amount"].corr(df_model["mrr_amount"])
    if abs(corr) > 0.999:
        DROP_FEATURES.append("arr_amount")
if DROP_FEATURES:
    df_model = df_model.drop(columns=DROP_FEATURES)

y_full = df_model["churn_flag"]
X_full = df_model.drop(columns=["churn_flag"])

_, _, _, _, train_idx, test_idx = train_test_split(
    X_full, y_full, X_full.index,
    test_size=TEST_SIZE, stratify=y_full, random_state=RANDOM_STATE
)

# Original unscaled data
df_orig = pd.read_csv(ORIGINAL_DATA_PATH)
orig_test = df_orig.loc[test_idx].copy().reset_index(drop=True)

# Fill missing values with median (same as eda_pipeline)
mrr_median = df_orig["mrr_amount"].median()
orig_test["mrr_amount"] = orig_test["mrr_amount"].fillna(mrr_median)

print(f"[OK] Original test sample data: {orig_test.shape}")
print(f"     MRR (original): mean=${orig_test['mrr_amount'].mean():,.0f}, "
      f"median=${orig_test['mrr_amount'].median():,.0f}")
print(f"     tenure_days: mean={orig_test['tenure_days'].mean():.0f} days")


# =============================================================================
# STEP 2: XAI — SHAP INTERPRETATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: SHAP Feature Importance (TreeExplainer)")
print("=" * 70)

explainer = shap.TreeExplainer(model)
print("[OK] TreeExplainer initialized")

shap_values = explainer.shap_values(X_test)
print(f"[OK] SHAP values computed: shape={np.array(shap_values).shape}")

# --- 2.1 Summary plot ---
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
plt.title("SHAP Feature Importance (Top-20 Features)", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
shap_path = os.path.join(RESULTS_DIR, "shap_feature_importance.png")
plt.savefig(shap_path, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"[OK] SHAP summary plot saved: {shap_path}")

# --- 2.2 Beeswarm plot (detailed impact distribution) ---
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values, X_test, show=False, max_display=20)
plt.title("SHAP Beeswarm: Feature Impact on Churn", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
beeswarm_path = os.path.join(RESULTS_DIR, "shap_beeswarm.png")
plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"[OK] SHAP beeswarm plot saved: {beeswarm_path}")

# --- 2.3 Top-10 features by mean |SHAP| ---
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("\n[INFO] Top-10 features by mean |SHAP|:")
for i, row in feature_importance.head(10).iterrows():
    print(f"       {row['feature']:35s} {row['mean_abs_shap']:.4f}")


# =============================================================================
# STEP 3: FINANCIAL METRICS & LTV BASELINE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Financial Metrics -- LTV Baseline")
print("=" * 70)

# --- 3.1 Historical LTV = MRR * (tenure_days / 30) ---
orig_test["ltv"] = orig_test["mrr_amount"] * (orig_test["tenure_days"] / 30)

print(f"\n[INFO] Historical LTV (test sample, n={len(orig_test)}):")
print(f"       Mean LTV   = ${orig_test['ltv'].mean():>12,.2f}")
print(f"       Median LTV = ${orig_test['ltv'].median():>12,.2f}")
print(f"       Total LTV  = ${orig_test['ltv'].sum():>14,.2f}")
print(f"       Min LTV    = ${orig_test['ltv'].min():>12,.2f}")
print(f"       Max LTV    = ${orig_test['ltv'].max():>12,.2f}")

# --- 3.2 LTV by churn ---
y_test_reset = y_test.reset_index(drop=True)
orig_test["churn_flag"] = y_test_reset.values
orig_test["plan_tier"] = df_orig.loc[test_idx, "plan_tier"].values

ltv_by_churn = orig_test.groupby("churn_flag")["ltv"].agg(["mean", "median", "sum", "count"])
ltv_by_churn.columns = ["Mean LTV", "Median LTV", "Total LTV", "Count"]
print(f"\n[INFO] LTV by churn_flag:")
print(ltv_by_churn.to_string(float_format="${:>12,.2f}".format))


# =============================================================================
# STEP 4: BUSINESS ROI SIMULATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Business ROI Simulation")
print("=" * 70)

# --- 4.1 Model predictions on test sample ---
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test).astype(int)

orig_test["churn_prob"] = y_prob
orig_test["churn_pred"] = y_pred

total_clients = len(orig_test)
predicted_churners = orig_test[orig_test["churn_pred"] == 1].copy()
n_predicted_churn = len(predicted_churners)

print(f"\n[INFO] Total clients in test set: {total_clients}")
print(f"[INFO] Model predicted churn for: {n_predicted_churn} clients "
      f"({n_predicted_churn/total_clients:.1%})")

# --- 4.2 Top-20% highest risk ---
threshold_count = max(int(total_clients * TOP_RISK_PERCENTILE), 1)
top_risk = orig_test.nlargest(threshold_count, "churn_prob").copy()

print(f"\n[INFO] Top-{TOP_RISK_PERCENTILE:.0%} highest-risk clients: {len(top_risk)}")
print(f"       Min churn probability in this group: {top_risk['churn_prob'].min():.4f}")
print(f"       Max: {top_risk['churn_prob'].max():.4f}")
print(f"       Mean: {top_risk['churn_prob'].mean():.4f}")

# --- 4.3 MRR at risk ---
mrr_at_risk = top_risk["mrr_amount"].sum()
print(f"\n{'=' * 50}")
print(f"  TOTAL MRR AT RISK")
print(f"  {len(top_risk)} clients x avg ${top_risk['mrr_amount'].mean():,.0f}/mo")
print(f"  = ${mrr_at_risk:>14,.2f} / month")
print(f"{'=' * 50}")

# --- 4.4 Retention strategy: 20% discount, 40% retention ---
n_retained = int(len(top_risk) * RETENTION_RATE)
retained_clients = top_risk.nlargest(n_retained, "mrr_amount")  # Retain highest-value clients

mrr_retained_gross = retained_clients["mrr_amount"].sum()
discount_cost = mrr_retained_gross * DISCOUNT_RATE
mrr_retained_net = mrr_retained_gross - discount_cost

# Clients not retained
n_lost = len(top_risk) - n_retained
lost_clients = top_risk.drop(retained_clients.index)
mrr_lost = lost_clients["mrr_amount"].sum()

print(f"\n{'=' * 50}")
print(f"  RETENTION STRATEGY")
print(f"{'=' * 50}")
print(f"  Client discount:               {DISCOUNT_RATE:.0%}")
print(f"  Retention rate:                {RETENTION_RATE:.0%}")
print(f"  Clients retained:              {n_retained} clients")
print(f"  Clients lost:                  {n_lost} clients")

print(f"\n  --- Monthly Financial Result ---")
print(f"  Gross MRR retained:           ${mrr_retained_gross:>12,.2f}")
print(f"  Discount cost (20%):          -${discount_cost:>11,.2f}")
print(f"  -----------------------------------------------")
print(f"  NET SAVED MRR:                ${mrr_retained_net:>12,.2f} / mo")
print(f"  MRR lost (not retained):      -${mrr_lost:>11,.2f} / mo")

# --- 4.5 Annual projection ---
annual_saved = mrr_retained_net * 12
annual_lost = mrr_lost * 12
annual_discount_cost = discount_cost * 12

print(f"\n  --- Annual Projection (x12 months) ---")
print(f"  Net saved ARR:                ${annual_saved:>14,.2f}")
print(f"  Annual discount cost:         ${annual_discount_cost:>14,.2f}")
print(f"  ARR lost:                     ${annual_lost:>14,.2f}")

# --- 4.6 Discount program ROI ---
if annual_discount_cost > 0:
    roi = (annual_saved / annual_discount_cost) * 100
    print(f"\n  Retention program ROI:        {roi:>10,.0f}%")
    print(f"  (Each $1 discount returns ${roi/100:,.2f})")

print(f"\n{'=' * 50}")


# =============================================================================
# STEP 5: EXPORT SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Exporting business summary report")
print("=" * 70)

# --- 5.1 Detailed at-risk client table ---
risk_report = top_risk[[
    "mrr_amount", "tenure_days", "ltv", "churn_prob", "churn_pred", "plan_tier"
]].copy()
risk_report = risk_report.sort_values("churn_prob", ascending=False)
risk_report_path = os.path.join(RESULTS_DIR, "top_risk_clients.csv")
risk_report.to_csv(risk_report_path, index=False)
print(f"[OK] Top-risk clients: {risk_report_path} ({len(risk_report)} rows)")

# --- 5.2 Summary metrics table ---
summary = {
    "Metric": [
        "Clients in test set",
        "Predicted churn (model)",
        "Top-20% at risk",
        "MRR at risk (monthly)",
        "Clients retained (40%)",
        "Gross MRR retained (monthly)",
        "Discount cost 20% (monthly)",
        "Net saved MRR (monthly)",
        "Net saved ARR (annual)",
        "MRR lost (monthly)",
        "Retention program ROI",
    ],
    "Value": [
        f"{total_clients}",
        f"{n_predicted_churn}",
        f"{len(top_risk)}",
        f"${mrr_at_risk:,.2f}",
        f"{n_retained}",
        f"${mrr_retained_gross:,.2f}",
        f"${discount_cost:,.2f}",
        f"${mrr_retained_net:,.2f}",
        f"${annual_saved:,.2f}",
        f"${mrr_lost:,.2f}",
        f"{roi:.0f}%" if annual_discount_cost > 0 else "N/A",
    ],
}
summary_df = pd.DataFrame(summary)
summary_path = os.path.join(RESULTS_DIR, "business_summary.csv")
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"[OK] Summary table: {summary_path}")

# --- 5.3 SHAP feature importance CSV ---
fi_path = os.path.join(RESULTS_DIR, "shap_feature_importance.csv")
feature_importance.to_csv(fi_path, index=False)
print(f"[OK] SHAP importance: {fi_path}")


# =============================================================================
# FINAL REPORT
# =============================================================================
print("\n" + "=" * 70)
print("PIPELINE COMPLETED")
print("=" * 70)
print(f"""
Artifacts:
  |-- {shap_path}
  |-- {beeswarm_path}
  |-- {risk_report_path}
  |-- {summary_path}
  +-- {fi_path}

Key business metrics:
  MRR at risk:                ${mrr_at_risk:>14,.2f} / mo
  Clients retained:            {n_retained} of {len(top_risk)}
  Net saved MRR:              ${mrr_retained_net:>14,.2f} / mo
  Net saved ARR:              ${annual_saved:>14,.2f} / yr
  Program ROI:                 {roi:.0f}%
""")
