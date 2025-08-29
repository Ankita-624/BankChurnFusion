import time, json, os, sys
import numpy as np
import pandas as pd
import joblib
import requests

ART = "artifacts"
API = "http://127.0.0.1:8000"

# --- load artifacts
scaler = joblib.load(f"{ART}/robust_scaler.joblib")
dt = joblib.load(f"{ART}/dt_hard.joblib")
lr = joblib.load(f"{ART}/lr_stacker_change.joblib")
THRESH = json.load(open(f"{ART}/threshold_change.json"))["threshold"]
WINSOR = json.load(open(f"{ART}/winsor_limits.json"))

def winsorize(v, low, high):
    return max(min(v, high), low)

def soft_expert(tenure, tx, bal, default_history):
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

def change_score(tenure, tx, bal):
    tx_prev  = tx * 0.95 or 1e-6
    bal_prev = bal * 0.95 or 1e-6
    ten_prev = max(tenure - 1, 0)
    dtx  = (tx  - tx_prev)  / tx_prev
    dbal = (bal - bal_prev) / bal_prev
    dten = tenure - ten_prev
    s = 0.0
    if dtx < -0.30: s += 0.50
    elif dtx < -0.15: s += 0.30
    if dbal < -0.25: s += 0.30
    elif dbal < -0.10: s += 0.15
    if dten <= 0: s += 0.20
    return min(s, 1.0)

def local_score(tenure, tx, bal, loan, cc, default):
    # winsorize + scale (use same feature names as training)
    tx_w  = winsorize(tx,  WINSOR["Transactions"]["low"], WINSOR["Transactions"]["high"])
    bal_w = winsorize(bal, WINSOR["AvgBalance"]["low"],  WINSOR["AvgBalance"]["high"])
    Z_in = pd.DataFrame({"Tenure":[tenure], "Transactions_w":[tx_w], "AvgBalance_w":[bal_w]})
    z = scaler.transform(Z_in)[0]
    z_t, z_x, z_b = z.tolist()
    # hard (named DF to match training)
    Xh = pd.DataFrame([{
        "z_Tenure": z_t, "z_Transactions": z_x, "z_AvgBalance": z_b,
        "LoanHistory": loan, "CreditCardUsage": cc, "DefaultHistory": default
    }])
    p_hard = float(dt.predict_proba(Xh)[:,1][0])
    # soft + change
    p_soft = float(soft_expert(tenure, tx, bal, default))
    cs = float(change_score(tenure, tx, bal))
    # fused
    pf = float(lr.predict_proba(np.array([[p_hard, p_soft, cs]]))[:,1][0])
    y = int(pf >= THRESH)
    return p_hard, p_soft, cs, pf, y

def normalize_payload_keys(payload):
    """Accept TitleCase from API examples; map to local_score args."""
    return dict(
        tenure = payload["Tenure"],
        tx     = payload["Transactions"],
        bal    = payload["AvgBalance"],
        loan   = payload["LoanHistory"],
        cc     = payload["CreditCardUsage"],
        default= payload["DefaultHistory"],
    )

def call_api(payload):
    r = requests.post(f"{API}/predict_churn", json=payload, timeout=5)
    r.raise_for_status()
    return r.json()

def approx(a,b,eps=1e-5): return abs(a-b) <= eps

def wait_for_health(max_wait=20):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            r = requests.get(f"{API}/health", timeout=2)
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(1)
    raise RuntimeError("API /health not reachable. Start the server in another terminal.")

def check_case(payload, name):
    args = normalize_payload_keys(payload)
    ph, ps, cs, pf, y = local_score(**args)
    resp = call_api(payload)
    ok = (
        approx(ph, resp["P_Hard"]) and
        approx(ps, resp["P_Soft"]) and
        approx(cs, resp["Change_Score"]) and
        approx(pf, resp["P_Fused"]) and
        (y == resp["Churn_Predicted"])
    )
    print(f"[{name}] {'OK' if ok else 'MISMATCH'} | "
          f"local: {ph:.6f}/{ps:.6f}/{cs:.3f}/{pf:.6f}/{y} | api: {resp}")
    return ok

if __name__ == "__main__":
    h = wait_for_health()
    print("Health:", h)

    risky = {"Tenure":1,"Transactions":25,"AvgBalance":80000,"LoanHistory":0,"CreditCardUsage":0,"DefaultHistory":1}
    safe  = {"Tenure":8,"Transactions":180,"AvgBalance":900000,"LoanHistory":1,"CreditCardUsage":1,"DefaultHistory":0}
    mid   = {"Tenure":5,"Transactions":70,"AvgBalance":350000,"LoanHistory":1,"CreditCardUsage":0,"DefaultHistory":0}

    ok = True
    ok &= check_case(risky, "risky")
    ok &= check_case(safe,  "safe")
    ok &= check_case(mid,   "mid")

    print("\nREADY TO DEPLOY ✅" if ok else "\nFIX MISMATCHES ❌")
    sys.exit(0 if ok else 1)
