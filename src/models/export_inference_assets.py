import os, json, numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import joblib

RANDOM_STATE = 42
os.makedirs("artifacts", exist_ok=True)

# ---------- Load data ----------
df_proc = pd.read_csv("data/bank_customers_processed.csv")   # has Tenure, Transactions_w, AvgBalance_w, z_* ...
df_soft = pd.read_csv("data/churn_soft_scores.csv")          # has P_Hard (from last run) + P_Soft + Churn

# ---------- Persist winsor limits on raw L/F/M ----------
# (We will apply these to incoming raw fields before scaling at inference)
winsor_limits = {
    "Transactions": {
        "low": float(df_proc["Transactions"].quantile(0.01)),
        "high": float(df_proc["Transactions"].quantile(0.99)),
    },
    "AvgBalance": {
        "low": float(df_proc["AvgBalance"].quantile(0.01)),
        "high": float(df_proc["AvgBalance"].quantile(0.99)),
    }
}
with open("artifacts/winsor_limits.json", "w") as f:
    json.dump(winsor_limits, f, indent=2)

# ---------- Fit RobustScaler on winsorized L/F/M ----------
scaler = RobustScaler()
Z = scaler.fit_transform(df_proc[["Tenure", "Transactions_w", "AvgBalance_w"]])
joblib.dump(scaler, "artifacts/robust_scaler.joblib")

# ---------- Train the hard decision tree (same config you used) ----------
X = pd.DataFrame(Z, columns=["z_Tenure","z_Transactions","z_AvgBalance"])
X["LoanHistory"] = df_proc["LoanHistory"].values
X["CreditCardUsage"] = df_proc["CreditCardUsage"].values
X["DefaultHistory"] = df_proc["DefaultHistory"].values
y = df_proc["Churn"].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)

classes = np.unique(y_tr)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
class_weights = dict(zip(classes, weights))

dt = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=7,
    min_samples_split=30,
    min_samples_leaf=15,
    class_weight=class_weights,
    random_state=RANDOM_STATE
)
dt.fit(X_tr, y_tr)
joblib.dump(dt, "artifacts/dt_hard.joblib")

# quick sanity
p_tr = dt.predict_proba(X_tr)[:,1]; p_te = dt.predict_proba(X_te)[:,1]
print("HARD ROC-AUC (valid):", roc_auc_score(y_te, p_te))

# ---------- Train the stacker over [P_Hard, P_Soft] ----------
assert {"P_Hard","P_Soft","Churn"}.issubset(df_soft.columns)
XS = df_soft[["P_Hard","P_Soft"]].values
yS = df_soft["Churn"].values

XS_tr, XS_te, yS_tr, yS_te = train_test_split(XS, yS, test_size=0.30, random_state=RANDOM_STATE, stratify=yS)
lr = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE)
lr.fit(XS_tr, yS_tr)
p_te_stack = lr.predict_proba(XS_te)[:,1]
print("STACKED ROC-AUC (valid):", roc_auc_score(yS_te, p_te_stack))

# threshold tuning
best_f1, best_t = -1, 0.5
for t in np.linspace(0.10, 0.90, 33):
    f1 = f1_score(yS_te, (p_te_stack >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t
print(f"Best threshold t={best_t:.2f}, F1={best_f1:.4f}")

joblib.dump(lr, "artifacts/lr_stacker.joblib")
with open("artifacts/threshold.json", "w") as f:
    json.dump({"threshold": best_t}, f)

print("✅ Saved artifacts in artifacts/: robust_scaler, winsor_limits, dt_hard, lr_stacker, threshold")
