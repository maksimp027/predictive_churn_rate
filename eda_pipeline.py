"""
EDA & Normalization Pipeline для SaaS Churn Prediction проєкту.
Виконує глибокий розвідувальний аналіз даних та підготовку до моделювання.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Без GUI — зберігаємо графіки на диск
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP & INGESTION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("КРОК 1: Завантаження даних та ініціалізація середовища")
print("=" * 60)

DATA_PATH = "data/saas_master_dataset_x6.csv"
PLOTS_DIR = "eda_plots"
OUTPUT_PATH = "data/model_ready_data.csv"

os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"[OK] Директорія для графіків: '{PLOTS_DIR}/'")

df = pd.read_csv(DATA_PATH)
print(f"[OK] Датасет завантажено: {df.shape[0]} рядків, {df.shape[1]} колонок")
print(f"\nКолонки:\n{df.columns.tolist()}")
print(f"\nТипи даних:\n{df.dtypes}")
print(f"\nПропущені значення:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nЦільова змінна — розподіл:\n{df['churn_flag'].value_counts()}")
print(f"Баланс класів: {df['churn_flag'].mean():.2%} відтік")

# ─────────────────────────────────────────────────────────────────────────────
# 2. VISUAL EDA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("КРОК 2: Візуальний EDA — генерація та збереження графіків")
print("=" * 60)

# Налаштування стилю
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
COLORS = ["#4C9BE8", "#E85C4C"]  # синій = лояльні, червоний = відтік

# --- Графік 1: Дисбаланс класів churn_flag ---
print("\n[1/4] Будуємо Bar plot дисбалансу класів...")
fig, ax = plt.subplots(figsize=(7, 5))
churn_counts = df["churn_flag"].value_counts().sort_index()
bars = ax.bar(
    ["Лояльні (0)", "Відтік (1)"],
    churn_counts.values,
    color=COLORS,
    edgecolor="white",
    linewidth=1.5,
    width=0.5,
)
# Підписи значень та відсотків над стовпцями
for bar, count in zip(bars, churn_counts.values):
    pct = count / len(df) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 15,
        f"{count:,}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
ax.set_title("Дисбаланс класів: churn_flag", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Кількість записів")
ax.set_ylim(0, churn_counts.max() * 1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
path1 = os.path.join(PLOTS_DIR, "01_class_imbalance.png")
plt.savefig(path1, dpi=150)
plt.close()
print(f"[OK] Збережено: {path1}")

# --- Графік 2: Розподіл MRR у розрізі відтоку ---
print("\n[2/4] Будуємо KDE plot розподілу mrr_amount за churn_flag...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# KDE
for churn_val, color, label in zip([0, 1], COLORS, ["Лояльні", "Відтік"]):
    subset = df[df["churn_flag"] == churn_val]["mrr_amount"].dropna()
    axes[0].hist(subset, bins=40, alpha=0.6, color=color, label=label, density=True)
    subset.plot.kde(ax=axes[0], color=color, linewidth=2)
axes[0].set_title("Розподіл MRR (KDE)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("mrr_amount ($)")
axes[0].set_ylabel("Щільність")
axes[0].legend()
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Boxplot
df_box = df[["mrr_amount", "churn_flag"]].copy()
df_box["Статус"] = df_box["churn_flag"].map({0: "Лояльні", 1: "Відтік"})
sns.boxplot(
    data=df_box,
    x="Статус",
    y="mrr_amount",
    hue="Статус",
    palette={"Лояльні": COLORS[0], "Відтік": COLORS[1]},
    ax=axes[1],
    width=0.45,
    linewidth=1.5,
    legend=False,
)
axes[1].set_title("Boxplot MRR за статусом клієнта", fontsize=13, fontweight="bold")
axes[1].set_xlabel("")
axes[1].set_ylabel("mrr_amount ($)")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("Фінансові метрики (MRR) у розрізі відтоку", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
path2 = os.path.join(PLOTS_DIR, "02_mrr_by_churn.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Збережено: {path2}")

# --- Графік 3: Кореляційна матриця числових ознак ---
print("\n[3/4] Будуємо кореляційну матрицю числових ознак...")
# Вибираємо лише числові колонки, виключаємо ідентифікатори
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "account_id"]

corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Верхній трикутник — зайвий
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 7},
    ax=ax,
    square=True,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Кореляційна матриця числових ознак", fontsize=15, fontweight="bold", pad=15)
plt.tight_layout()
path3 = os.path.join(PLOTS_DIR, "03_correlation_heatmap.png")
plt.savefig(path3, dpi=150)
plt.close()
print(f"[OK] Збережено: {path3}")

# --- Графік 4: Вплив саппорту на відтік ---
print("\n[4/4] Будуємо Scatter plot: support_tickets_count vs avg_satisfaction_score...")
fig, ax = plt.subplots(figsize=(10, 6))

for churn_val, color, label, marker in zip(
    [0, 1], COLORS, ["Лояльні (churn=0)", "Відтік (churn=1)"], ["o", "X"]
):
    subset = df[df["churn_flag"] == churn_val]
    ax.scatter(
        subset["support_tickets_count"],
        subset["avg_satisfaction_score"],
        c=color,
        label=label,
        alpha=0.45,
        s=40,
        marker=marker,
        edgecolors="none",
    )

ax.set_title(
    "Вплив підтримки на відтік: Кількість тікетів vs Задоволеність",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_xlabel("support_tickets_count (кількість тікетів)")
ax.set_ylabel("avg_satisfaction_score (середня оцінка задоволеності)")
ax.legend(markerscale=1.5, fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
path4 = os.path.join(PLOTS_DIR, "04_support_vs_satisfaction.png")
plt.savefig(path4, dpi=150)
plt.close()
print(f"[OK] Збережено: {path4}")

print(f"\n[OK] Всі 4 графіки збережено у директорію '{PLOTS_DIR}/'")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEXT FEATURE ENGINEERING (NLP Baseline)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("КРОК 3: Text Feature Engineering (NLP Baseline)")
print("=" * 60)

# Колонка last_support_text відсутня в даному датасеті.
# Замість цього, витягуємо аналогічні базові ознаки з наявних текстових/категоріальних полів.
# reason_code — текстова причина відтоку/взаємодії.
if "last_support_text" in df.columns:
    print("[INFO] Знайдено колонку 'last_support_text' — обробляємо...")
    df["text_length"] = df["last_support_text"].fillna("").apply(len)
    df["has_ticket"] = (df["text_length"] > 0).astype(int)
    print(f"[OK] text_length: mid={df['text_length'].median():.0f}, max={df['text_length'].max()}")
    print(f"[OK] has_ticket: {df['has_ticket'].sum()} записів мають текст тікету")
else:
    print("[INFO] Колонка 'last_support_text' відсутня у цьому датасеті.")
    print("[INFO] Генеруємо аналогічні ознаки з доступних полів:")

    # text_length — довжина reason_code як проксі
    df["text_length"] = df["reason_code"].fillna("").apply(len)

    # has_ticket — клієнт має хоча б 1 тікет підтримки
    df["has_ticket"] = (df["support_tickets_count"].fillna(0) > 0).astype(int)

    print(f"  - text_length (len of reason_code): mid={df['text_length'].median():.0f}, max={df['text_length'].max()}")
    print(f"  - has_ticket (support_tickets_count > 0): {df['has_ticket'].sum()} з {len(df)} клієнтів")

# ─────────────────────────────────────────────────────────────────────────────
# 4. NORMALIZATION & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("КРОК 4: Нормалізація та препроцесинг")
print("=" * 60)

# --- 4.1 Відділяємо цільову змінну ---
TARGET = "churn_flag"
DROP_COLS = ["account_id"]  # Ідентифікатор — не ознака

print(f"\n[INFO] Цільова змінна: '{TARGET}'")
print(f"[INFO] Видаляємо технічні колонки: {DROP_COLS}")

y = df[TARGET].copy()
X = df.drop(columns=[TARGET] + DROP_COLS)
print(f"[OK] Форма X до обробки: {X.shape}")
print(f"[OK] Форма y: {y.shape}, розподіл: {y.value_counts().to_dict()}")

# --- 4.2 Визначаємо типи ознак ---
# Категоріальні: рядкові колонки, крім reason_code (вже оброблено через text_length)
CAT_COLS = X.select_dtypes(include=["str", "object"]).columns.tolist()
# reason_code — вже перетворено, можна дропнути
if "reason_code" in CAT_COLS:
    CAT_COLS.remove("reason_code")

# Числові: всі int/float
NUM_COLS = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n[INFO] Категоріальні ознаки ({len(CAT_COLS)}): {CAT_COLS}")
print(f"[INFO] Числові ознаки ({len(NUM_COLS)}): {NUM_COLS}")

# --- 4.3 Обробка пропущених значень ---
print("\n[INFO] Заповнення пропущених значень...")
missing_before = X[NUM_COLS].isnull().sum().sum()
X[NUM_COLS] = X[NUM_COLS].fillna(X[NUM_COLS].median())
X[CAT_COLS] = X[CAT_COLS].fillna("Unknown")
# reason_code — дропаємо (вже в text_length)
if "reason_code" in X.columns:
    X = X.drop(columns=["reason_code"])
print(f"[OK] Заповнено {missing_before} пропущених числових значень медіанами")

# --- 4.4 StandardScaler для числових ознак (для LogisticRegression baseline) ---
print("\n[INFO] Застосовуємо StandardScaler до числових колонок...")
# Оновлюємо список числових після дропу
num_cols_current = [c for c in NUM_COLS if c in X.columns]
scaler = StandardScaler()
X[num_cols_current] = scaler.fit_transform(X[num_cols_current])
print(f"[OK] Масштабовано {len(num_cols_current)} числових ознак")
print(f"     Приклад: mrr_amount -> mean={X['mrr_amount'].mean():.4f}, std={X['mrr_amount'].std():.4f}")

# --- 4.5 OneHotEncoding категоріальних ознак ---
print("\n[INFO] Застосовуємо pd.get_dummies до категоріальних ознак...")
cat_cols_current = [c for c in CAT_COLS if c in X.columns]
print(f"     Категоріальні для OHE: {cat_cols_current}")

X_encoded = pd.get_dummies(X, columns=cat_cols_current, drop_first=True, dtype=int)
print(f"[OK] Форма X після OHE: {X_encoded.shape}")
print(f"     Нових колонок додано: {X_encoded.shape[1] - X.shape[1] + len(cat_cols_current)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("КРОК 5: Експорт обробленого датасету")
print("=" * 60)

# Додаємо цільову змінну назад
model_ready = X_encoded.copy()
model_ready[TARGET] = y.values

os.makedirs("data", exist_ok=True)
model_ready.to_csv(OUTPUT_PATH, index=False)

print(f"[OK] Датасет збережено: '{OUTPUT_PATH}'")
print(f"     Фінальна форма: {model_ready.shape[0]} рядків x {model_ready.shape[1]} колонок")
print(f"     Колонки ({model_ready.shape[1]}): {model_ready.columns.tolist()[:10]} ...")
print(f"     Пропущені значення: {model_ready.isnull().sum().sum()}")

print("\n" + "=" * 60)
print("[OK] EDA Pipeline завершено успішно!")
print("=" * 60)
