"""
Train & Evaluation Pipeline для SaaS Churn Prediction.
Навчає 3 моделі (Baseline LogReg, CatBoost, XGBoost), порівнює метрики,
зберігає Champion Model та тестову вибірку для подальшого SHAP/ROI аналізу.
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

# --- Конфігурація ---

matplotlib.use("Agg")  # Без GUI — зберігаємо графіки на диск
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = "model_ready_data.csv"
RESULTS_DIR = "model_results"
TEST_SPLIT_DIR = os.path.join("data", "test_split")
RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEST_SPLIT_DIR, exist_ok=True)


# =============================================================================
# КРОК 1: ЗАВАНТАЖЕННЯ ДАНИХ ТА СТРАТИФІКОВАНИЙ SPLIT
# =============================================================================
print("=" * 70)
print("КРОК 1: Завантаження даних та стратифікований Train/Test split")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"[OK] Датасет завантажено: {df.shape[0]} рядків x {df.shape[1]} колонок")

# --- 1.1 Видалення ознак із нульовою дисперсією та мультиколінеарних ---
DROP_FEATURES = []

# has_open_ticket — нульова дисперсія (усі значення = 0)
if "has_open_ticket" in df.columns and df["has_open_ticket"].nunique() <= 1:
    DROP_FEATURES.append("has_open_ticket")

# arr_amount — точна копія mrr_amount × 12 (ідеальна кореляція)
if "arr_amount" in df.columns and "mrr_amount" in df.columns:
    corr = df["arr_amount"].corr(df["mrr_amount"])
    if abs(corr) > 0.999:
        DROP_FEATURES.append("arr_amount")

if DROP_FEATURES:
    df = df.drop(columns=DROP_FEATURES)
    print(f"[INFO] Видалено проблемні ознаки: {DROP_FEATURES}")
    print(f"       Форма після очищення: {df.shape}")

# --- 1.2 Відділення цільової змінної ---
TARGET = "churn_flag"
y = df[TARGET].copy()
X = df.drop(columns=[TARGET])

print(f"\n[INFO] Цільова змінна: '{TARGET}'")
print(f"       Ознаки (X): {X.shape[1]} колонок")
print(f"       Розподіл y: {y.value_counts().to_dict()}")
print(f"       Частка відтоку: {y.mean():.2%}")

# --- 1.3 Стратифікований split 80/20 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"\n[OK] Train/Test split ({1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}):")
print(f"     Train: {X_train.shape[0]} рядків | churn={y_train.mean():.2%}")
print(f"     Test:  {X_test.shape[0]} рядків | churn={y_test.mean():.2%}")


# =============================================================================
# КРОК 2: BASELINE — LOGISTIC REGRESSION
# =============================================================================
print("\n" + "=" * 70)
print("КРОК 2: Baseline — LogisticRegression (class_weight='balanced')")
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

print("[OK] LogisticRegression навчено.")
print(f"     ROC-AUC = {roc_auc_score(y_test, y_prob_lr):.4f}")


# =============================================================================
# КРОК 3: CATBOOST
# =============================================================================
print("\n" + "=" * 70)
print("КРОК 3: CatBoostClassifier (auto_class_weights='Balanced')")
print("=" * 70)

cb_model = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    auto_class_weights="Balanced",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=100,  # Лог кожні 100 ітерацій
)
cb_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

y_pred_cb = cb_model.predict(X_test).astype(int)
y_prob_cb = cb_model.predict_proba(X_test)[:, 1]

best_iter = cb_model.get_best_iteration()
print(f"\n[OK] CatBoost навчено. Найкраща ітерація: {best_iter}")
print(f"     ROC-AUC = {roc_auc_score(y_test, y_prob_cb):.4f}")


# =============================================================================
# КРОК 4: XGBOOST
# =============================================================================
print("\n" + "=" * 70)
print("КРОК 4: XGBClassifier (scale_pos_weight)")
print("=" * 70)

# Розрахунок scale_pos_weight: n_negative / n_positive
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

print(f"\n[OK] XGBoost навчено.")
print(f"     ROC-AUC = {roc_auc_score(y_test, y_prob_xgb):.4f}")


# =============================================================================
# КРОК 5: ПОРІВНЯЛЬНА ТАБЛИЦЯ МЕТРИК
# =============================================================================
print("\n" + "=" * 70)
print("КРОК 5: Порівняння моделей на тестовій вибірці")
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

# --- Визначення Champion Model за ROC-AUC ---
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

# --- Classification Report для Champion Model ---
print(f"\nClassification Report ({champion_name}):")
print(classification_report(y_test, champion_y_pred, target_names=["Лояльний", "Відтік"]))


# =============================================================================
# КРОК 6: ЗБЕРЕЖЕННЯ ГРАФІКІВ
# =============================================================================
print("=" * 70)
print("КРОК 6: Збереження графіків (Confusion Matrix + ROC-криві)")
print("=" * 70)

# --- 6.1 Confusion Matrix для Champion Model ---
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    champion_y_pred,
    display_labels=["Лояльний (0)", "Відтік (1)"],
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
print(f"[OK] Confusion Matrix збережено: {cm_path}")

# --- 6.2 ROC-криві всіх трьох моделей ---
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
ax.set_title("ROC-криві: порівняння моделей", fontsize=14, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(RESULTS_DIR, "roc_curves_comparison.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"[OK] ROC-криві збережено: {roc_path}")


# =============================================================================
# КРОК 7: СЕРІАЛІЗАЦІЯ МОДЕЛІ ТА ТЕСТОВОЇ ВИБІРКИ
# =============================================================================
print("\n" + "=" * 70)
print("КРОК 7: Серіалізація Champion Model та тестової вибірки")
print("=" * 70)

# --- 7.1 Збереження Champion Model ---
model_path = os.path.join(RESULTS_DIR, "champion_churn_model.pkl")

if champion_name == "CatBoost":
    # CatBoost має власний серіалізатор, але pkl теж підтримується через joblib
    cb_native_path = os.path.join(RESULTS_DIR, "champion_churn_model.cbm")
    champion_model.save_model(cb_native_path)
    print(f"[OK] CatBoost нативний формат: {cb_native_path}")

# Універсальний pkl для будь-якої моделі
joblib.dump(champion_model, model_path)
print(f"[OK] Champion Model ({champion_name}) збережено: {model_path}")

# --- 7.2 Збереження тестової вибірки для SHAP / ROI ---
X_test.to_csv(os.path.join(TEST_SPLIT_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(TEST_SPLIT_DIR, "y_test.csv"), index=False)
print(f"[OK] Тестова вибірка збережена: {TEST_SPLIT_DIR}/")
print(f"     X_test: {X_test.shape}")
print(f"     y_test: {y_test.shape}")

# --- 7.3 Збереження таблиці метрик ---
metrics_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
metrics_df.to_csv(metrics_path)
print(f"[OK] Таблиця метрик збережена: {metrics_path}")


# =============================================================================
# ФІНАЛЬНИЙ ЗВІТ
# =============================================================================
print("\n" + "=" * 70)
print("PIPELINE ЗАВЕРШЕНО")
print("=" * 70)
print(f"""
Артефакти:
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
