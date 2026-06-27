import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from catboost import CatBoostClassifier

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "model_ready_data.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── 1. Load data ────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['churn_flag'].mean():.2%}")

X = df.drop(columns=["churn_flag"])
y = df["churn_flag"]

# ── 2. Train / Test split 80/20 ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ── 3. Baseline — Logistic Regression ───────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

lr_proba = lr.predict_proba(X_test)[:, 1]
lr_pred  = lr.predict(X_test)

lr_auc = roc_auc_score(y_test, lr_proba)
lr_f1  = f1_score(y_test, lr_pred)
print(f"\n[Logistic Regression] ROC-AUC: {lr_auc:.4f} | F1: {lr_f1:.4f}")

# ── 4. CatBoost ─────────────────────────────────────────────────────────
cat = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    random_seed=42,
    verbose=100          # лог кожні 100 ітерацій
)
cat.fit(X_train, y_train, eval_set=(X_test, y_test))

cat_proba = cat.predict_proba(X_test)[:, 1]
cat_pred  = cat.predict(X_test)

cat_auc = roc_auc_score(y_test, cat_proba)
cat_f1  = f1_score(y_test, cat_pred)
print(f"\n[CatBoost]           ROC-AUC: {cat_auc:.4f} | F1: {cat_f1:.4f}")

# ── 5. Comparison table ─────────────────────────────────────────────────
print("\n" + "="*45)
print(f"{'Model':<25} {'ROC-AUC':>8} {'F1':>8}")
print("-"*45)
print(f"{'Logistic Regression':<25} {lr_auc:>8.4f} {lr_f1:>8.4f}")
print(f"{'CatBoost':<25} {cat_auc:>8.4f} {cat_f1:>8.4f}")
print("="*45)

# ── 6. Confusion Matrices + metrics table ───────────────────────────────
fig = plt.figure(figsize=(12, 8))
fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.01)

# Матриці
for idx, (preds, proba, title) in enumerate([
    (lr_pred,  lr_proba,  "Logistic Regression"),
    (cat_pred, cat_proba, "CatBoost")
], 1):
    ax = fig.add_subplot(2, 2, idx)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=12, fontweight="bold")

# Таблиця метрик внизу
ax_table = fig.add_subplot(2, 1, 2)
ax_table.axis("off")

table_data = [
    ["Метрика", "Logistic Regression", "CatBoost", "Переможець"],
    ["ROC-AUC", f"{lr_auc:.4f}", f"{cat_auc:.4f}", "🏆 CatBoost" if cat_auc > lr_auc else "🏆 LR"],
    ["F1-score", f"{lr_f1:.4f}", f"{cat_f1:.4f}", "🏆 CatBoost" if cat_f1 > lr_f1 else "🏆 LR"],
    ["Пропущених churn (FN)", str(confusion_matrix(y_test, lr_pred)[1][0]),
                              str(confusion_matrix(y_test, cat_pred)[1][0]),
                              "🏆 CatBoost"],
    ["Хибна тривога (FP)",    str(confusion_matrix(y_test, lr_pred)[0][1]),
                              str(confusion_matrix(y_test, cat_pred)[0][1]),
                              "🏆 CatBoost"],
]

table = ax_table.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Стилізація
for j in range(4):
    table[0, j].set_facecolor("#1F4E79")
    table[0, j].set_text_props(color="white", fontweight="bold")

for i in range(1, 5):
    for j in range(4):
        table[i, j].set_facecolor("#EBF3FB" if i % 2 == 0 else "white")

plt.tight_layout()
save_path = OUTPUTS_DIR / "model_comparison.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nГрафік збережено → {save_path}")
plt.show()
# ── 7. Save models ──────────────────────────────────────────────────────
joblib.dump(lr, MODELS_DIR / "logistic_regression.pkl")
cat.save_model(str(MODELS_DIR / "catboost_model.cbm"))
print("Models saved.")