import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression

# ----------------------------
# CONFIG
# ----------------------------
DATA_IN = "data/churn_soft_scores.csv"
DATA_OUT = "data/churn_final_fused.csv"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
RANDOM_STATE = 42

# ----------------------------
# 1) LOAD
# ----------------------------
df = pd.read_csv(DATA_IN)
print("Dataset shape:", df.shape)
print(df[["CustomerID", "P_Hard", "P_Soft"]].head())

# ----------------------------
# 2) BASELINE METRICS
# ----------------------------
roc_hard = roc_auc_score(df["Churn"], df["P_Hard"])
roc_soft = roc_auc_score(df["Churn"], df["P_Soft"])
print(f"Baseline ROC-AUC  P_Hard={roc_hard:.4f}  P_Soft={roc_soft:.4f}")

# ----------------------------
# 3) LEARNED FUSION (STACKING VIA LOGISTIC REGRESSION)
# ----------------------------
X = df[["P_Hard","P_Soft"]].values
y = df["Churn"].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

# simple, interpretable stacker
lr = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE)
lr.fit(X_tr, y_tr)

p_stack_tr = lr.predict_proba(X_tr)[:,1]
p_stack_te = lr.predict_proba(X_te)[:,1]

roc_stack = roc_auc_score(y_te, p_stack_te)
print(f"Stacked ROC-AUC (valid) = {roc_stack:.4f}")

# ----------------------------
# 4) THRESHOLD TUNING FOR CHURN (maximize F1 on validation)
# ----------------------------
best_f1, best_t = -1, 0.5
for t in np.linspace(0.10, 0.90, 33):
    pred = (p_stack_te >= t).astype(int)
    f1 = f1_score(y_te, pred)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"Best threshold by F1 on validation: t={best_t:.2f}, F1={best_f1:.4f}")

# Evaluate at best threshold
pred_best = (p_stack_te >= best_t).astype(int)
print("\n📊 Classification Report (Stacked @ tuned threshold):")
print(classification_report(y_te, pred_best, digits=4))

cm = confusion_matrix(y_te, pred_best)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"], ax=ax)
ax.set_title("Confusion Matrix - Fused (tuned)")
plt.savefig(f"{REPORTS_DIR}/06_confusion_matrix_fused.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ----------------------------
# 5) APPLY STACKER TO FULL DATA, SAVE
# ----------------------------
df["P_Fused"] = lr.predict_proba(X)[:,1]
df["Churn_Predicted"] = (df["P_Fused"] >= best_t).astype(int)

df.to_csv(DATA_OUT, index=False)
print(f"\n✅ Final fused churn probabilities saved to: {DATA_OUT}")
print(df[["CustomerID","P_Hard","P_Soft","P_Fused","Churn_Predicted"]].head())

# ----------------------------
# 6) DIAGNOSTIC PLOT
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["P_Hard"].values[:500], label="P_Hard", alpha=0.7)
plt.plot(df["P_Soft"].values[:500], label="P_Soft", alpha=0.7)
plt.plot(df["P_Fused"].values[:500], label="P_Fused (stacked)", linewidth=2, alpha=0.9)
plt.title("Churn Probabilities (first 500 customers)")
plt.xlabel("Customer Index")
plt.ylabel("Probability")
plt.legend()
plt.savefig(f"{REPORTS_DIR}/07_probabilities_fusion.png", bbox_inches="tight", dpi=150)
plt.close()
