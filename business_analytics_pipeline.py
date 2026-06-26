"""
Business Analytics Pipeline: XAI (SHAP) + Financial ROI Simulation.
Зв'язує технічні передбачення Champion Model з реальними грошима.
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

# --- Конфiгурацiя ---

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

# Бiзнес-параметри ROI-симуляцiї
DISCOUNT_RATE = 0.20       # 20% знижка для утримання
RETENTION_RATE = 0.40      # 40% клiєнтiв погоджуються залишитися
TOP_RISK_PERCENTILE = 0.20 # Топ-20% з найвищим ризиком

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# КРОК 1: ЗАВАНТАЖЕННЯ АРТЕФАКТIВ
# =============================================================================
print("=" * 70)
print("КРОК 1: Завантаження моделi, тестових данних та оригiнального датасету")
print("=" * 70)

# --- 1.1 CatBoost модель ---
model = CatBoostClassifier()
model.load_model(MODEL_PATH)
print(f"[OK] Модель завантажено: {MODEL_PATH}")
print(f"     Кiлькiсть дерев: {model.tree_count_}")

# --- 1.2 Тестовi сплiти (масштабованi) ---
X_test = pd.read_csv(TEST_X_PATH)
y_test = pd.read_csv(TEST_Y_PATH).squeeze()
print(f"[OK] X_test: {X_test.shape}")
print(f"[OK] y_test: {y_test.shape}, churn={y_test.mean():.2%}")

# --- 1.3 Вiдтворюємо iндекси test-split для доступу до оригiнальних значень ---
# Повторюємо точно той самий split, що був у train_pipeline.py
df_model = pd.read_csv(MODEL_READY_PATH)

# Видаляємо тi самi колонки, що й у train_pipeline
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

# Оригiнальнi немасштабованi данi
df_orig = pd.read_csv(ORIGINAL_DATA_PATH)
orig_test = df_orig.loc[test_idx].copy().reset_index(drop=True)

# Заповнюємо пропуски медiаною (як у eda_pipeline)
mrr_median = df_orig["mrr_amount"].median()
orig_test["mrr_amount"] = orig_test["mrr_amount"].fillna(mrr_median)

print(f"[OK] Оригiнальнi данi тест-вибiрки: {orig_test.shape}")
print(f"     MRR (оригiнальний): mean=${orig_test['mrr_amount'].mean():,.0f}, "
      f"median=${orig_test['mrr_amount'].median():,.0f}")
print(f"     tenure_days: mean={orig_test['tenure_days'].mean():.0f} днiв")


# =============================================================================
# КРОК 2: XAI — SHAP INTERPRETATION
# =============================================================================
print("\n" + "=" * 70)
print("КРОК 2: SHAP Feature Importance (TreeExplainer)")
print("=" * 70)

explainer = shap.TreeExplainer(model)
print("[OK] TreeExplainer iнiцiалiзовано")

shap_values = explainer.shap_values(X_test)
print(f"[OK] SHAP values розраховано: shape={np.array(shap_values).shape}")

# --- 2.1 Summary plot ---
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
plt.title("SHAP Feature Importance (Top-20 ознак)", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
shap_path = os.path.join(RESULTS_DIR, "shap_feature_importance.png")
plt.savefig(shap_path, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"[OK] SHAP summary plot збережено: {shap_path}")

# --- 2.2 Beeswarm plot (детальний розподiл впливу) ---
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values, X_test, show=False, max_display=20)
plt.title("SHAP Beeswarm: вплив ознак на вiдтiк", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
beeswarm_path = os.path.join(RESULTS_DIR, "shap_beeswarm.png")
plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"[OK] SHAP beeswarm plot збережено: {beeswarm_path}")

# --- 2.3 Топ-10 ознак за середнiм |SHAP| ---
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("\n[INFO] Топ-10 ознак за середнiм |SHAP|:")
for i, row in feature_importance.head(10).iterrows():
    print(f"       {row['feature']:35s} {row['mean_abs_shap']:.4f}")


# =============================================================================
# КРОК 3: FINANCIAL METRICS & LTV BASELINE
# =============================================================================
print("\n" + "=" * 70)
print("КРОК 3: Financial Metrics -- LTV Baseline")
print("=" * 70)

# --- 3.1 Iсторичний LTV = MRR * (tenure_days / 30) ---
orig_test["ltv"] = orig_test["mrr_amount"] * (orig_test["tenure_days"] / 30)

print(f"\n[INFO] Iсторичний LTV (тест-вибiрка, n={len(orig_test)}):")
print(f"       Mean LTV   = ${orig_test['ltv'].mean():>12,.2f}")
print(f"       Median LTV = ${orig_test['ltv'].median():>12,.2f}")
print(f"       Total LTV  = ${orig_test['ltv'].sum():>14,.2f}")
print(f"       Min LTV    = ${orig_test['ltv'].min():>12,.2f}")
print(f"       Max LTV    = ${orig_test['ltv'].max():>12,.2f}")

# --- 3.2 LTV у розрiзi churn ---
y_test_reset = y_test.reset_index(drop=True)
orig_test["churn_flag"] = y_test_reset.values
orig_test["plan_tier"] = df_orig.loc[test_idx, "plan_tier"].values

ltv_by_churn = orig_test.groupby("churn_flag")["ltv"].agg(["mean", "median", "sum", "count"])
ltv_by_churn.columns = ["Mean LTV", "Median LTV", "Total LTV", "Count"]
print(f"\n[INFO] LTV у розрiзi churn_flag:")
print(ltv_by_churn.to_string(float_format="${:>12,.2f}".format))


# =============================================================================
# КРОК 4: BUSINESS ROI SIMULATION
# =============================================================================
print("\n" + "=" * 70)
print("КРОК 4: Business ROI Simulation (грошi для Домiнiки)")
print("=" * 70)

# --- 4.1 Прогнози моделi на тест-вибiрцi ---
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test).astype(int)

orig_test["churn_prob"] = y_prob
orig_test["churn_pred"] = y_pred

total_clients = len(orig_test)
predicted_churners = orig_test[orig_test["churn_pred"] == 1].copy()
n_predicted_churn = len(predicted_churners)

print(f"\n[INFO] Загальна кiлькiсть клiєнтiв у тестi: {total_clients}")
print(f"[INFO] Модель передбачила вiдтiк для: {n_predicted_churn} клiєнтiв "
      f"({n_predicted_churn/total_clients:.1%})")

# --- 4.2 Топ-20% з найвищим ризиком ---
threshold_count = max(int(total_clients * TOP_RISK_PERCENTILE), 1)
top_risk = orig_test.nlargest(threshold_count, "churn_prob").copy()

print(f"\n[INFO] Топ-{TOP_RISK_PERCENTILE:.0%} клiєнтiв з найвищим ризиком: {len(top_risk)}")
print(f"       Мiнiмальна ймовiрнiсть вiдтоку в цiй групi: {top_risk['churn_prob'].min():.4f}")
print(f"       Максимальна: {top_risk['churn_prob'].max():.4f}")
print(f"       Середня:     {top_risk['churn_prob'].mean():.4f}")

# --- 4.3 MRR пiд загрозою ---
mrr_at_risk = top_risk["mrr_amount"].sum()
print(f"\n{'=' * 50}")
print(f"  ЗАГАЛЬНИЙ MRR ПIД ЗАГРОЗОЮ ВТРАТИ")
print(f"  {len(top_risk)} клiєнтiв x avg ${top_risk['mrr_amount'].mean():,.0f}/мiс")
print(f"  = ${mrr_at_risk:>14,.2f} / мiсяць")
print(f"{'=' * 50}")

# --- 4.4 Стратегiя утримання: 20% знижка, 40% retention ---
n_retained = int(len(top_risk) * RETENTION_RATE)
retained_clients = top_risk.nlargest(n_retained, "mrr_amount")  # Утримуємо найдорожчих

mrr_retained_gross = retained_clients["mrr_amount"].sum()
discount_cost = mrr_retained_gross * DISCOUNT_RATE
mrr_retained_net = mrr_retained_gross - discount_cost

# Клiєнти, яких не вдалося утримати
n_lost = len(top_risk) - n_retained
lost_clients = top_risk.drop(retained_clients.index)
mrr_lost = lost_clients["mrr_amount"].sum()

print(f"\n{'=' * 50}")
print(f"  СТРАТЕГIЯ УТРИМАННЯ")
print(f"{'=' * 50}")
print(f"  Знижка для клiєнтiв:          {DISCOUNT_RATE:.0%}")
print(f"  Коефiцiєнт утримання:         {RETENTION_RATE:.0%}")
print(f"  Кiлькiсть утриманих:           {n_retained} клiєнтiв")
print(f"  Кiлькiсть втрачених:           {n_lost} клiєнтiв")

print(f"\n  --- Фiнансовий результат (щомiсячний) ---")
print(f"  Валовий MRR утриманих:        ${mrr_retained_gross:>12,.2f}")
print(f"  Витрати на знижки (20%):      -${discount_cost:>11,.2f}")
print(f"  -----------------------------------------------")
print(f"  ЧИСТИЙ ЗБЕРЕЖЕНИЙ MRR:        ${mrr_retained_net:>12,.2f} / мiс")
print(f"  MRR втрачений (не утриманi):  -${mrr_lost:>11,.2f} / мiс")

# --- 4.5 Рiчний ефект ---
annual_saved = mrr_retained_net * 12
annual_lost = mrr_lost * 12
annual_discount_cost = discount_cost * 12

print(f"\n  --- Рiчна проекцiя (x12 мiсяцiв) ---")
print(f"  Чистий збережений ARR:        ${annual_saved:>14,.2f}")
print(f"  Рiчнi витрати на знижки:      ${annual_discount_cost:>14,.2f}")
print(f"  ARR втрачений:                ${annual_lost:>14,.2f}")

# --- 4.6 ROI знижкової програми ---
if annual_discount_cost > 0:
    roi = (annual_saved / annual_discount_cost) * 100
    print(f"\n  ROI програми утримання:       {roi:>10,.0f}%")
    print(f"  (На кожен $1 знижки повертається ${roi/100:,.2f})")

print(f"\n{'=' * 50}")


# =============================================================================
# КРОК 5: ЕКСПОРТ ЗВЕДЕНОГО ЗВIТУ
# =============================================================================
print("\n" + "=" * 70)
print("КРОК 5: Експорт зведеного бiзнес-звiту")
print("=" * 70)

# --- 5.1 Детальна таблиця клiєнтiв пiд ризиком ---
risk_report = top_risk[[
    "mrr_amount", "tenure_days", "ltv", "churn_prob", "churn_pred", "plan_tier"
]].copy()
risk_report = risk_report.sort_values("churn_prob", ascending=False)
risk_report_path = os.path.join(RESULTS_DIR, "top_risk_clients.csv")
risk_report.to_csv(risk_report_path, index=False)
print(f"[OK] Топ-ризик клiєнти: {risk_report_path} ({len(risk_report)} рядкiв)")

# --- 5.2 Зведена таблиця метрик ---
summary = {
    "Метрика": [
        "Клiєнтiв у тестi",
        "Передбачено вiдтiк (модель)",
        "Топ-20% пiд ризиком",
        "MRR пiд загрозою (мiс.)",
        "Утримано клiєнтiв (40%)",
        "Валовий MRR утриманих (мiс.)",
        "Витрати на знижки 20% (мiс.)",
        "Чистий збережений MRR (мiс.)",
        "Чистий збережений ARR (рiк)",
        "MRR втрачений (мiс.)",
        "ROI програми утримання",
    ],
    "Значення": [
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
print(f"[OK] Зведена таблиця: {summary_path}")

# --- 5.3 SHAP feature importance CSV ---
fi_path = os.path.join(RESULTS_DIR, "shap_feature_importance.csv")
feature_importance.to_csv(fi_path, index=False)
print(f"[OK] SHAP importance: {fi_path}")


# =============================================================================
# ФIНАЛЬНИЙ ЗВIТ
# =============================================================================
print("\n" + "=" * 70)
print("PIPELINE ЗАВЕРШЕНО")
print("=" * 70)
print(f"""
Артефакти:
  |-- {shap_path}
  |-- {beeswarm_path}
  |-- {risk_report_path}
  |-- {summary_path}
  +-- {fi_path}

Ключовi бiзнес-цифри для Домiнiки:
  MRR пiд загрозою:          ${mrr_at_risk:>14,.2f} / мiс
  Утримано клiєнтiв:          {n_retained} з {len(top_risk)}
  Чистий збережений MRR:     ${mrr_retained_net:>14,.2f} / мiс
  Чистий збережений ARR:     ${annual_saved:>14,.2f} / рiк
  ROI програми:               {roi:.0f}%
""")
