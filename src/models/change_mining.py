import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

RANDOM_STATE = 42

DATA_IN = "data/bank_customers_processed.csv"   # has Churn
SOFT_IN = "data/churn_soft_scores.csv"          # has P_Hard, P_Soft (maybe Churn, but we won't rely on it)
DATA_OUT = "data/churn_change_scores.csv"
ART_DIR = "artifacts"

os.makedirs(ART_DIR, exist_ok=True)

# ----------------------------
# 1) Load processed base (source of truth for Churn)
# ----------------------------
df = pd.read_csv(DATA_IN)
print("Base processed shape:", df.shape)
if "Churn" not in df.columns:
    raise RuntimeError("Expected 'Churn' column in bank_customers_processed.csv")

# ----------------------------
# 2) Simulate a previous-period snapshot and compute deltas
# ----------------------------
np.random.seed(RANDOM_STATE)

# Simulate previous period (light noise); clip to avoid zeros/negatives
df["Transactions_prev"] = (df["Transactions"] * np.random.uniform(0.85, 1.10, size=len(df))).clip(lower=1)
df["AvgBalance_prev"]  = (df["AvgBalance"]  * np.random.uniform(0.85, 1.10, size=len(df))).clip(lower=1)
df["Tenure_prev"]      = np.maximum(df["Tenure"] - np.random.randint(0, 2, size=len(df)), 0)

df["Delta_Tenure"] = df["Tenure"] - df["Tenure_prev"]
df["Delta_Tx"]     = (df["Transactions"] - df["Transactions_prev"]) / df["Transactions_prev"]
df["Delta_Bal"]    = (df["AvgBalance"]   - df["AvgBalance_prev"])  / df["AvgBalance_prev"]

# ----------------------------
# 3) Change score (higher for negative shifts)
# ----------------------------
def compute_change_score(row):
    score = 0.0
    # Transactions drop
    if row["Delta_Tx"] < -0.30:
        score += 0.50
    elif row["Delta_Tx"] < -0.15:
        score += 0.30
    # Balance drop
    if row["Delta_Bal"] < -0.25:
        score += 0.30
    elif row["Delta_Bal"] < -0.10:
        score += 0.15
    # Tenure stagnates or drops (no growth)
    if row["Delta_Tenure"] <= 0:
        score += 0.20
    return min(score, 1.0)

df["change_score"] = df.apply(compute_change_score, axis=1)

# ----------------------------
# 4) Bring in P_Hard / P_Soft (do NOT pull Churn from soft file)
# ----------------------------
soft = pd.read_csv(SOFT_IN)
cols = ["CustomerID"]
for c in ["P_Hard", "P_Soft"]:
    if c not in soft.columns:
        raise RuntimeError(f"Expected '{c}' in {SOFT_IN}")
    cols.append(c)

soft = soft[cols]
df = df.merge(soft, on="CustomerID", how="left")

# sanity fill if any missing after merge (shouldn't happen)
df["P_Hard"] = df["P_Hard"].fillna(0.0)
df["P_Soft"] = df["P_Soft"].fillna(df["P_Hard"] * 0.5)  # weak fallback

# ----------------------------
# 5) Train updated stacker on [P_Hard, P_Soft, change_score]
# ----------------------------
X = df[["P_Hard", "P_Soft", "change_score"]].values
y = df["Churn"].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

lr = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE)
lr.fit(X_tr, y_tr)

p_te = lr.predict_proba(X_te)[:, 1]
roc = roc_auc_score(y_te, p_te)

best_f1, best_t = -1.0, 0.5
for t in np.linspace(0.10, 0.90, 33):
    f1 = f1_score(y_te, (p_te >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"\n📊 Updated Stacker ROC-AUC: {roc:.4f}")
print(f"Best Threshold = {best_t:.2f} | Best F1 = {best_f1:.4f}")

# ----------------------------
# 6) Save artifacts + per-customer scores
# ----------------------------
joblib.dump(lr, os.path.join(ART_DIR, "lr_stacker_change.joblib"))
with open(os.path.join(ART_DIR, "threshold_change.json"), "w") as f:
    json.dump({"threshold": float(best_t)}, f)

out_cols = ["CustomerID", "P_Hard", "P_Soft", "change_score", "Churn",
            "Delta_Tx", "Delta_Bal", "Delta_Tenure"]
df[out_cols].to_csv(DATA_OUT, index=False)
print(f"\n✅ Change-mining scores saved → {DATA_OUT}")
print(df[out_cols].head())
