"""
EDA & Normalization Pipeline for SaaS Churn Prediction project.
Performs deep exploratory data analysis and prepares data for modeling.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # No GUI — save plots to disk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP & INGESTION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data and initializing environment")
print("=" * 60)

DATA_PATH = "data/saas_master_dataset_x6.csv"
PLOTS_DIR = "eda_plots"
OUTPUT_PATH = "model_ready_data.csv"

os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"[OK] Plots directory: '{PLOTS_DIR}/'")

df = pd.read_csv(DATA_PATH)
print(f"[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nTarget variable — distribution:\n{df['churn_flag'].value_counts()}")
print(f"Class balance: {df['churn_flag'].mean():.2%} churn")

# ─────────────────────────────────────────────────────────────────────────────
# 2. VISUAL EDA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Visual EDA — generating and saving plots")
print("=" * 60)

# Style configuration
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
COLORS = ["#4C9BE8", "#E85C4C"]  # blue = retained, red = churned

# --- Plot 1: churn_flag class imbalance ---
print("\n[1/4] Building class imbalance bar plot...")
fig, ax = plt.subplots(figsize=(7, 5))
churn_counts = df["churn_flag"].value_counts().sort_index()
bars = ax.bar(
    ["Retained (0)", "Churned (1)"],
    churn_counts.values,
    color=COLORS,
    edgecolor="white",
    linewidth=1.5,
    width=0.5,
)
# Value and percentage labels above bars
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
ax.set_title("Class Imbalance: churn_flag", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Record Count")
ax.set_ylim(0, churn_counts.max() * 1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
path1 = os.path.join(PLOTS_DIR, "01_class_imbalance.png")
plt.savefig(path1, dpi=150)
plt.close()
print(f"[OK] Saved: {path1}")

# --- Plot 2: MRR distribution by churn ---
print("\n[2/4] Building KDE plot of mrr_amount by churn_flag...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# KDE
for churn_val, color, label in zip([0, 1], COLORS, ["Retained", "Churned"]):
    subset = df[df["churn_flag"] == churn_val]["mrr_amount"].dropna()
    axes[0].hist(subset, bins=40, alpha=0.6, color=color, label=label, density=True)
    subset.plot.kde(ax=axes[0], color=color, linewidth=2)
axes[0].set_title("MRR Distribution (KDE)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("mrr_amount ($)")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Boxplot
df_box = df[["mrr_amount", "churn_flag"]].copy()
df_box["Status"] = df_box["churn_flag"].map({0: "Retained", 1: "Churned"})
sns.boxplot(
    data=df_box,
    x="Status",
    y="mrr_amount",
    hue="Status",
    palette={"Retained": COLORS[0], "Churned": COLORS[1]},
    ax=axes[1],
    width=0.45,
    linewidth=1.5,
    legend=False,
)
axes[1].set_title("MRR Boxplot by Customer Status", fontsize=13, fontweight="bold")
axes[1].set_xlabel("")
axes[1].set_ylabel("mrr_amount ($)")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("Financial Metrics (MRR) by Churn Status", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
path2 = os.path.join(PLOTS_DIR, "02_mrr_by_churn.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Saved: {path2}")

# --- Plot 3: Correlation matrix of numeric features ---
print("\n[3/4] Building numeric feature correlation matrix...")
# Select only numeric columns, exclude identifiers
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "account_id"]

corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle — redundant
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
ax.set_title("Correlation Matrix of Numeric Features", fontsize=15, fontweight="bold", pad=15)
plt.tight_layout()
path3 = os.path.join(PLOTS_DIR, "03_correlation_heatmap.png")
plt.savefig(path3, dpi=150)
plt.close()
print(f"[OK] Saved: {path3}")

# --- Plot 4: Support impact on churn ---
print("\n[4/4] Building scatter plot: support_tickets_count vs avg_satisfaction_score...")
fig, ax = plt.subplots(figsize=(10, 6))

for churn_val, color, label, marker in zip(
    [0, 1], COLORS, ["Retained (churn=0)", "Churned (churn=1)"], ["o", "X"]
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
    "Support Impact on Churn: Ticket Count vs Satisfaction",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_xlabel("support_tickets_count (number of tickets)")
ax.set_ylabel("avg_satisfaction_score (mean satisfaction rating)")
ax.legend(markerscale=1.5, fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
path4 = os.path.join(PLOTS_DIR, "04_support_vs_satisfaction.png")
plt.savefig(path4, dpi=150)
plt.close()
print(f"[OK] Saved: {path4}")

print(f"\n[OK] All 4 plots saved to '{PLOTS_DIR}/'")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEXT FEATURE ENGINEERING (NLP Baseline)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Text Feature Engineering (NLP Baseline)")
print("=" * 60)

# Column last_support_text is absent in this dataset.
# Instead, extract analogous baseline features from available text/categorical fields.
# reason_code — textual churn/interaction reason.
if "last_support_text" in df.columns:
    print("[INFO] Found column 'last_support_text' — processing...")
    df["text_length"] = df["last_support_text"].fillna("").apply(len)
    df["has_ticket"] = (df["text_length"] > 0).astype(int)
    print(f"[OK] text_length: mid={df['text_length'].median():.0f}, max={df['text_length'].max()}")
    print(f"[OK] has_ticket: {df['has_ticket'].sum()} records have ticket text")
else:
    print("[INFO] Column 'last_support_text' not found in this dataset.")
    print("[INFO] Generating analogous features from available fields:")

    # text_length — length of reason_code as a proxy
    df["text_length"] = df["reason_code"].fillna("").apply(len)

    # has_ticket — customer has at least 1 support ticket
    df["has_ticket"] = (df["support_tickets_count"].fillna(0) > 0).astype(int)

    print(f"  - text_length (len of reason_code): mid={df['text_length'].median():.0f}, max={df['text_length'].max()}")
    print(f"  - has_ticket (support_tickets_count > 0): {df['has_ticket'].sum()} of {len(df)} customers")

# ─────────────────────────────────────────────────────────────────────────────
# 4. NORMALIZATION & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Normalization and preprocessing")
print("=" * 60)

# --- 4.1 Separate target variable ---
TARGET = "churn_flag"
DROP_COLS = ["account_id"]  # Identifier — not a feature

print(f"\n[INFO] Target variable: '{TARGET}'")
print(f"[INFO] Dropping technical columns: {DROP_COLS}")

y = df[TARGET].copy()
X = df.drop(columns=[TARGET] + DROP_COLS)
print(f"[OK] X shape before processing: {X.shape}")
print(f"[OK] y shape: {y.shape}, distribution: {y.value_counts().to_dict()}")

# --- 4.2 Identify feature types ---
# Categorical: string columns, except reason_code (already processed via text_length)
CAT_COLS = X.select_dtypes(include=["str", "object"]).columns.tolist()
# reason_code — already converted, can be dropped
if "reason_code" in CAT_COLS:
    CAT_COLS.remove("reason_code")

# Numeric: all int/float
NUM_COLS = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n[INFO] Categorical features ({len(CAT_COLS)}): {CAT_COLS}")
print(f"[INFO] Numeric features ({len(NUM_COLS)}): {NUM_COLS}")

# --- 4.3 Handle missing values ---
print("\n[INFO] Filling missing values...")
missing_before = X[NUM_COLS].isnull().sum().sum()
X[NUM_COLS] = X[NUM_COLS].fillna(X[NUM_COLS].median())
X[CAT_COLS] = X[CAT_COLS].fillna("Unknown")
# reason_code — drop (already captured in text_length)
if "reason_code" in X.columns:
    X = X.drop(columns=["reason_code"])
print(f"[OK] Filled {missing_before} missing numeric values with medians")

# --- 4.4 StandardScaler for numeric features (for LogisticRegression baseline) ---
print("\n[INFO] Applying StandardScaler to numeric columns...")
# Update numeric column list after drops
num_cols_current = [c for c in NUM_COLS if c in X.columns]
scaler = StandardScaler()
X[num_cols_current] = scaler.fit_transform(X[num_cols_current])
print(f"[OK] Scaled {len(num_cols_current)} numeric features")
print(f"     Example: mrr_amount -> mean={X['mrr_amount'].mean():.4f}, std={X['mrr_amount'].std():.4f}")

# --- 4.5 OneHotEncoding of categorical features ---
print("\n[INFO] Applying pd.get_dummies to categorical features...")
cat_cols_current = [c for c in CAT_COLS if c in X.columns]
print(f"     Categorical for OHE: {cat_cols_current}")

X_encoded = pd.get_dummies(X, columns=cat_cols_current, drop_first=True, dtype=int)
print(f"[OK] X shape after OHE: {X_encoded.shape}")
print(f"     New columns added: {X_encoded.shape[1] - X.shape[1] + len(cat_cols_current)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Exporting processed dataset")
print("=" * 60)

# Add target variable back
model_ready = X_encoded.copy()
model_ready[TARGET] = y.values

os.makedirs("data", exist_ok=True)
model_ready.to_csv(OUTPUT_PATH, index=False)

print(f"[OK] Dataset saved: '{OUTPUT_PATH}'")
print(f"     Final shape: {model_ready.shape[0]} rows x {model_ready.shape[1]} columns")
print(f"     Columns ({model_ready.shape[1]}): {model_ready.columns.tolist()[:10]} ...")
print(f"     Missing values: {model_ready.isnull().sum().sum()}")

print("\n" + "=" * 60)
print("[OK] EDA Pipeline completed successfully!")
print("=" * 60)
