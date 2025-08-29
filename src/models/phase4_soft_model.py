import os
import numpy as np
import pandas as pd
import random

# ----------------------------
# CONFIG
# ----------------------------
DATA_IN = "data/churn_hard_scores.csv"
DATA_OUT = "data/churn_soft_scores.csv"
os.makedirs("data", exist_ok=True)
random.seed(42)
np.random.seed(42)

# ----------------------------
# 1) LOAD
# ----------------------------
df = pd.read_csv(DATA_IN)
print("Dataset shape:", df.shape)
print(df.head(3))

# ----------------------------
# 2) EXPERT SIMULATION (stronger, more discriminative)
# ----------------------------
def simulate_expert_opinion(row):
    """
    Returns dict: {"Churn": p, "NonChurn": p, "Uncertain": p}
    Heuristics:
      - short tenure, low balance, low transactions, prior default -> higher churn
      - long tenure, high balance, high transactions -> lower churn
    """
    # Base priors (less biased to NonChurn)
    p_churn = 0.30
    p_non   = 0.50
    p_unc   = 0.20

    # VERY HIGH RISK
    if (row["AvgBalance"] < 150_000) and (row["Transactions"] < 40) and (row["Tenure"] < 3):
        p_churn, p_non, p_unc = 0.85, 0.10, 0.05

    # HIGH RISK
    elif (row["AvgBalance"] < 300_000) and (row["Transactions"] < 60):
        p_churn, p_non, p_unc = 0.65, 0.20, 0.15

    # Default history tilts further to churn
    if row.get("DefaultHistory", 0) == 1:
        p_churn = min(p_churn + 0.10, 0.95)
        p_non   = max(p_non - 0.08, 0.0)
        p_unc   = max(0.0, 1 - p_churn - p_non)

    # STRONG RETENTION
    if (row["AvgBalance"] > 700_000) and (row["Transactions"] > 150) and (row["Tenure"] > 6):
        p_churn, p_non, p_unc = 0.03, 0.92, 0.05

    # small disagreement noise
    noise = random.uniform(-0.06, 0.06)
    p_churn = min(max(p_churn + noise, 0), 1)
    p_non   = min(max(p_non - noise/2, 0), 1)
    p_unc   = max(0.0, 1 - p_churn - p_non)

    return {"Churn": p_churn, "NonChurn": p_non, "Uncertain": p_unc}

# ----------------------------
# 3) DEMPSTER-SHAFER COMBINATION (manual)
# ----------------------------
def combine_evidence(m1, m2):
    """
    Dempster's rule for 3 masses: Churn, NonChurn, Uncertain.
    m1, m2: dicts with keys "Churn","NonChurn","Uncertain"
    """
    k = m1["Churn"] * m2["NonChurn"] + m1["NonChurn"] * m2["Churn"]
    if k >= 0.999999:  # near total conflict
        return {"Churn": 0.0, "NonChurn": 0.0, "Uncertain": 1.0}

    churn = (m1["Churn"] * m2["Churn"]
             + m1["Churn"] * m2["Uncertain"]
             + m1["Uncertain"] * m2["Churn"]) / (1 - k)

    non   = (m1["NonChurn"] * m2["NonChurn"]
             + m1["NonChurn"] * m2["Uncertain"]
             + m1["Uncertain"] * m2["NonChurn"]) / (1 - k)

    unc   = (m1["Uncertain"] * m2["Uncertain"]) / (1 - k)

    s = churn + non + unc
    if s > 0:
        churn, non, unc = churn/s, non/s, unc/s
    return {"Churn": churn, "NonChurn": non, "Uncertain": unc}

# ----------------------------
# 4) BUILD SOFT SCORES
# ----------------------------
# Fewer experts to avoid over-confident collapse
expert_counts = 3
soft_scores = []

cols_needed = ["Tenure","Transactions","AvgBalance","DefaultHistory"]
subset = df[cols_needed].to_dict(orient="records")

for row_vals in subset:
    combined = simulate_expert_opinion(row_vals)
    for _ in range(expert_counts - 1):
        next_op = simulate_expert_opinion(row_vals)
        combined = combine_evidence(combined, next_op)

    # expected churn probability: belief + half of uncertainty
    p_soft = combined["Churn"] + 0.5 * combined["Uncertain"]
    soft_scores.append(p_soft)

df["P_Soft"] = soft_scores

# ----------------------------
# 5) SAVE
# ----------------------------
df.to_csv(DATA_OUT, index=False)
print(f"✅ Soft churn probabilities saved to: {DATA_OUT}")
print(df[["CustomerID","P_Hard","P_Soft"]].head(10))
