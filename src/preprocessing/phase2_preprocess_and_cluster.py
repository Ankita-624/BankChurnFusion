import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------------
# Config
# --------------------
DATA_IN = "data/bank_customers.csv"
DATA_OUT = "data/bank_customers_processed.csv"
FIG_DIR = "reports/figures"
RANDOM_STATE = 42

# --------------------
# Utilities
# --------------------
def ensure_dirs():
    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

def winsorize_series(s, lower_q=0.01, upper_q=0.99):
    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    return s.clip(lower=lower, upper=upper)

def label_clusters_by_intensity(df, cluster_col, features, label_col="LFM_Band"):
    """
    Map numeric clusters to human labels (Low/Medium/High) by ranking clusters
    on the mean of (z_Tenure + z_Transactions + z_AvgBalance).
    """
    z = df[[f"z_{c}" for c in features]]
    df["_intensity"] = z.sum(axis=1)
    order = (
        df.groupby(cluster_col)["_intensity"]
          .mean()
          .sort_values()
          .index.tolist()
    )
    # Smallest intensity => 'Low', middle => 'Medium', largest => 'High'
    mapping = {}
    if len(order) == 2:
        mapping = {order[0]:"Low", order[1]:"High"}
    elif len(order) == 3:
        mapping = {order[0]:"Low", order[1]:"Medium", order[2]:"High"}
    else:
        # For k>3, distribute labels; still keeps an ordered notion
        names = ["Very Low","Low","Medium","High","Very High","Ultra"]
        mapping = {cid:names[i] if i < len(names) else f"Band{i}" for i, cid in enumerate(order)}
    df[label_col] = df[cluster_col].map(mapping)
    df.drop(columns=["_intensity"], inplace=True)
    return mapping

def plot_and_save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

# --------------------
# 1) Load
# --------------------
ensure_dirs()
df = pd.read_csv(DATA_IN)

# Basic sanity
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head(3))

# --------------------
# 2) Quick EDA snapshots (saved as PNGs)
# --------------------
# Churn distribution
fig, ax = plt.subplots()
df["Churn"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_title("Churn class distribution (0=No, 1=Yes)")
ax.set_xlabel("Churn")
ax.set_ylabel("Count")
plot_and_save(fig, "01_churn_distribution.png")

# Histograms for L, F, M
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
df["Tenure"].plot(kind="hist", bins=30, ax=axes[0]); axes[0].set_title("Tenure (years)")
df["Transactions"].plot(kind="hist", bins=30, ax=axes[1]); axes[1].set_title("Transactions/year")
df["AvgBalance"].plot(kind="hist", bins=30, ax=axes[2]); axes[2].set_title("Average Balance")
plot_and_save(fig, "02_LFM_hist.png")

# Boxplots for outlier sense check
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
sns.boxplot(y=df["Tenure"], ax=axes[0]); axes[0].set_title("Tenure")
sns.boxplot(y=df["Transactions"], ax=axes[1]); axes[1].set_title("Transactions")
sns.boxplot(y=df["AvgBalance"], ax=axes[2]); axes[2].set_title("AvgBalance")
plot_and_save(fig, "03_LFM_box.png")

# --------------------
# 3) Handle outliers (winsorize Transactions + AvgBalance)
# --------------------
df["Transactions_w"] = winsorize_series(df["Transactions"], 0.01, 0.99)
df["AvgBalance_w"] = winsorize_series(df["AvgBalance"], 0.01, 0.99)

# --------------------
# 4) Scale L, F, M (RobustScaler handles outliers better)
# --------------------
features = ["Tenure", "Transactions_w", "AvgBalance_w"]
scaler = RobustScaler()
Z = scaler.fit_transform(df[features])
Z = pd.DataFrame(Z, columns=[f"z_{c.split('_')[0]}" for c in features])  # z_Tenure, z_Transactions, z_AvgBalance
df = pd.concat([df, Z], axis=1)

# --------------------
# 5) Choose k via silhouette on (z_Tenure, z_Transactions, z_AvgBalance)
# --------------------
X = df[[c for c in df.columns if c.startswith("z_")]].values

sil_scores = {}
for k in range(2, 7):  # 2..6
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores[k] = float(score)
    print(f"k={k} silhouette={score:.4f}")

# Plot silhouette scores
fig, ax = plt.subplots()
ax.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o")
ax.set_xlabel("k")
ax.set_ylabel("Silhouette score")
ax.set_title("Silhouette sweep for k-means on L/F/M (z-scaled)")
plot_and_save(fig, "04_silhouette_k_sweep.png")

best_k = max(sil_scores, key=sil_scores.get)
print("Best k by silhouette:", best_k)

# --------------------
# 6) Final k-means fit
# --------------------
km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init="auto")
df["LFM_Cluster"] = km_final.fit_predict(X)

# --------------------
# 7) Human-friendly labels (Low / Medium / High ...)
# --------------------
mapping = label_clusters_by_intensity(df, "LFM_Cluster", features=["Tenure","Transactions","AvgBalance"])
print("Cluster → Band mapping:", mapping)

# Save mapping for later steps
with open("data/cluster_label_mapping.json", "w") as f:
    json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)

# --------------------
# 8) Visual sanity: bands by churn rate
# --------------------
band_order = df["LFM_Band"].value_counts().sort_index().index.tolist()
churn_by_band = df.groupby("LFM_Band")["Churn"].mean().reindex(band_order)

fig, ax = plt.subplots()
churn_by_band.plot(kind="bar", ax=ax)
ax.set_title("Churn rate by LFM band")
ax.set_xlabel("LFM band")
ax.set_ylabel("Mean churn")
plot_and_save(fig, "05_churn_by_band.png")

# --------------------
# 9) Save processed dataset
# --------------------
# Keep the winsorized and z-scored features + cluster/band
out_cols = [
    "CustomerID","Age","Gender","AnnualIncome","Tenure","Transactions","Transactions_w",
    "AvgBalance","AvgBalance_w","LoanHistory","CreditCardUsage","DefaultHistory","Churn",
    "z_Tenure","z_Transactions","z_AvgBalance","LFM_Cluster","LFM_Band"
]
df[out_cols].to_csv(DATA_OUT, index=False)
print(f"[OK] Saved processed dataset → {DATA_OUT}")
print(df[out_cols].head())
