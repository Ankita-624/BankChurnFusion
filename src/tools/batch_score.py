import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

ART_DIR = "artifacts"

# --- Load artifacts (change-aware) ---
scaler = joblib.load(f"{ART_DIR}/robust_scaler.joblib")
dt = joblib.load(f"{ART_DIR}/dt_hard.joblib")
lr = joblib.load(f"{ART_DIR}/lr_stacker_change.joblib")
THRESH = json.load(open(f"{ART_DIR}/threshold_change.json"))["threshold"]
WINSOR = json.load(open(f"{ART_DIR}/winsor_limits.json"))

# --- Helpers ---
def winsorize(val, low, high):
    return max(min(val, high), low)

def soft_expert(tenure, tx, bal, default_history):
    # Deterministic expert heuristic
    p_churn, p_non, p_unc = 0.30, 0.50, 0.20
    if (bal < 150_000) and (tx < 40) and (tenure < 3):
        p_churn, p_non, p_unc = 0.85, 0.10, 0.05
    elif (bal < 300_000) and (tx < 60):
        p_churn, p_non, p_unc = 0.65, 0.20, 0.15
    if default_history == 1:
        p_churn = min(p_churn + 0.10, 0.95)
        p_non   = max(p_non - 0.08, 0.0)
        p_unc   = max(0.0, 1 - p_churn - p_non)
    if (bal > 700_000) and (tx > 150) and (tenure > 6):
        p_churn, p_non, p_unc = 0.03, 0.92, 0.05
    return p_churn + 0.5 * p_unc

def compute_change_score(tenure, tx, bal):
    # Deterministic previous period
    tx_prev = tx * 0.95
    bal_prev = bal * 0.95
    ten_prev = max(tenure - 1, 0)

    # Avoid zero divisions
    tx_prev = tx_prev if tx_prev != 0 else 1e-6
    bal_prev = bal_prev if bal_prev != 0 else 1e-6

    delta_tx = (tx - tx_prev) / tx_prev
    delta_bal = (bal - bal_prev) / bal_prev
    delta_ten = tenure - ten_prev

    score = 0.0
    if delta_tx < -0.30: score += 0.50
    elif delta_tx < -0.15: score += 0.30
    if delta_bal < -0.25: score += 0.30
    elif delta_bal < -0.10: score += 0.15
    if delta_ten <= 0: score += 0.20
    return min(score, 1.0)

def score_df(df_in: pd.DataFrame) -> pd.DataFrame:
    # Required columns
    req = ["CustomerID","Tenure","Transactions","AvgBalance","LoanHistory","CreditCardUsage","DefaultHistory"]
    missing = [c for c in req if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df_in.copy()

    # ---- Winsorize L/F/M with training limits ----
    df["Transactions_w"] = df["Transactions"].apply(lambda v: winsorize(v, WINSOR["Transactions"]["low"], WINSOR["Transactions"]["high"]))
    df["AvgBalance_w"]   = df["AvgBalance"].apply(lambda v: winsorize(v, WINSOR["AvgBalance"]["low"], WINSOR["AvgBalance"]["high"]))

    # ---- Scale (keep feature names used during training) ----
    Z_in = pd.DataFrame({
        "Tenure": df["Tenure"].values,
        "Transactions_w": df["Transactions_w"].values,
        "AvgBalance_w": df["AvgBalance_w"].values
    })
    Z = scaler.transform(Z_in)
    Z_df = pd.DataFrame(Z, columns=["z_Tenure","z_Transactions","z_AvgBalance"])

    # ---- Hard model features ----
    X_hard_df = pd.DataFrame({
        "z_Tenure": Z_df["z_Tenure"].values,
        "z_Transactions": Z_df["z_Transactions"].values,
        "z_AvgBalance": Z_df["z_AvgBalance"].values,
        "LoanHistory": df["LoanHistory"].values,
        "CreditCardUsage": df["CreditCardUsage"].values,
        "DefaultHistory": df["DefaultHistory"].values
    })
    p_hard = dt.predict_proba(X_hard_df)[:, 1]

    # ---- Soft + Change scores ----
    p_soft = np.array([
        soft_expert(t, x, b, d)
        for t, x, b, d in zip(df["Tenure"], df["Transactions"], df["AvgBalance"], df["DefaultHistory"])
    ])
    change_scores = np.array([
        compute_change_score(t, x, b)
        for t, x, b in zip(df["Tenure"], df["Transactions"], df["AvgBalance"])
    ])

    # ---- Fused prediction (3 features) ----
    X_stack = np.column_stack((p_hard, p_soft, change_scores))
    p_fused = lr.predict_proba(X_stack)[:, 1]
    y_hat = (p_fused >= THRESH).astype(int)

    # ---- Output ----
    out = df_in.copy()
    out["P_Hard"] = p_hard
    out["P_Soft"] = p_soft
    out["Change_Score"] = change_scores
    out["P_Fused"] = p_fused
    out["Churn_Predicted"] = y_hat
    out["Rank"] = out["P_Fused"].rank(method="first", ascending=False).astype(int)
    out.sort_values("P_Fused", ascending=False, inplace=True)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV with required columns")
    ap.add_argument("--out", dest="out", default="data/scored_customers.csv", help="Output CSV path")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    scored = score_df(df)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    scored.to_csv(args.out, index=False)
    print(f"Saved: {args.out}  (rows={len(scored)})")
