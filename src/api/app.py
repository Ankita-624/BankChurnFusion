from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib, json, numpy as np, pandas as pd, io

app = FastAPI(
    title="Bank Churn Fusion API",
    version="2.1",
    description="Hard + Soft + Change-Mining + Fusion-based churn scoring with batch upload"
)

from fastapi.middleware.cors import CORSMiddleware

ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://bank-churn-dashboard.netlify.app",  # your Netlify URL (exact)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],      # lets preflight OPTIONS pass
    allow_headers=["*"],      # allows Content-Type: multipart/form-data, etc.
)


# --- artifacts ---
scaler = joblib.load("artifacts/robust_scaler.joblib")
dt = joblib.load("artifacts/dt_hard.joblib")
lr = joblib.load("artifacts/lr_stacker_change.joblib")
WINSOR = json.load(open("artifacts/winsor_limits.json"))
THRESH = json.load(open("artifacts/threshold_change.json"))["threshold"]

# --- utils shared by /predict_churn and /batch_score ---
def winsorize(val, low, high): return max(min(val, high), low)

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

def compute_change_score(tenure, tx, bal):
    tx_prev = tx * 0.95 or 1e-6
    bal_prev = bal * 0.95 or 1e-6
    ten_prev = max(tenure - 1, 0)
    dtx  = (tx - tx_prev) / tx_prev
    dbal = (bal - bal_prev) / bal_prev
    dten = tenure - ten_prev
    s = 0.0
    if dtx < -0.30: s += 0.50
    elif dtx < -0.15: s += 0.30
    if dbal < -0.25: s += 0.30
    elif dbal < -0.10: s += 0.15
    if dten <= 0: s += 0.20
    return min(s, 1.0)

def score_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    req = ["CustomerID","Tenure","Transactions","AvgBalance","LoanHistory","CreditCardUsage","DefaultHistory"]
    missing = [c for c in req if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df_in.copy()

    # winsor + scale
    df["Transactions_w"] = df["Transactions"].apply(lambda v: winsorize(v, WINSOR["Transactions"]["low"], WINSOR["Transactions"]["high"]))
    df["AvgBalance_w"]   = df["AvgBalance"].apply(lambda v: winsorize(v, WINSOR["AvgBalance"]["low"], WINSOR["AvgBalance"]["high"]))
    Z = scaler.transform(pd.DataFrame({
        "Tenure": df["Tenure"].values,
        "Transactions_w": df["Transactions_w"].values,
        "AvgBalance_w": df["AvgBalance_w"].values
    }))
    Z_df = pd.DataFrame(Z, columns=["z_Tenure","z_Transactions","z_AvgBalance"])

    # hard model
    Xh = pd.DataFrame({
        "z_Tenure": Z_df["z_Tenure"].values,
        "z_Transactions": Z_df["z_Transactions"].values,
        "z_AvgBalance": Z_df["z_AvgBalance"].values,
        "LoanHistory": df["LoanHistory"].values,
        "CreditCardUsage": df["CreditCardUsage"].values,
        "DefaultHistory": df["DefaultHistory"].values
    })
    p_hard = dt.predict_proba(Xh)[:,1]

    # soft + change
    p_soft = np.array([
        soft_expert(t,x,b,d)
        for t,x,b,d in zip(df["Tenure"], df["Transactions"], df["AvgBalance"], df["DefaultHistory"])
    ])
    change_scores = np.array([
        compute_change_score(t,x,b)
        for t,x,b in zip(df["Tenure"], df["Transactions"], df["AvgBalance"])
    ])

    # fusion
    Xs = np.column_stack((p_hard, p_soft, change_scores))
    p_fused = lr.predict_proba(Xs)[:,1]
    yhat = (p_fused >= THRESH).astype(int)

    out = df[["CustomerID"]].copy()
    out["P_Hard"] = p_hard
    out["P_Soft"] = p_soft
    out["Change_Score"] = change_scores
    out["P_Fused"] = p_fused
    out["Churn_Predicted"] = yhat
    return out

# --- schemas & endpoints ---
class Customer(BaseModel):
    Tenure: float = Field(..., ge=0)
    Transactions: float = Field(..., ge=0)
    AvgBalance: float = Field(..., ge=0)
    LoanHistory: int = Field(..., ge=0, le=1)
    CreditCardUsage: int = Field(..., ge=0, le=1)
    DefaultHistory: int = Field(..., ge=0, le=1)

@app.get("/")
def home():
    return {"message": "Bank Churn Fusion API v2.1 → Visit /docs for Swagger UI"}

@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESH, "stacker_model": "lr_stacker_change.joblib", "message": "Artifacts loaded successfully"}

@app.post("/predict_churn")
def predict_churn(customer: Customer):
    tx_w  = winsorize(customer.Transactions, WINSOR["Transactions"]["low"], WINSOR["Transactions"]["high"])
    bal_w = winsorize(customer.AvgBalance,  WINSOR["AvgBalance"]["low"],  WINSOR["AvgBalance"]["high"])
    z = scaler.transform([[customer.Tenure, tx_w, bal_w]])[0]
    z_t, z_x, z_b = z.tolist()
    Xh = np.array([[z_t, z_x, z_b, customer.LoanHistory, customer.CreditCardUsage, customer.DefaultHistory]])
    p_hard = float(dt.predict_proba(Xh)[:,1][0])
    p_soft = float(soft_expert(customer.Tenure, customer.Transactions, customer.AvgBalance, customer.DefaultHistory))
    change_score = float(compute_change_score(customer.Tenure, customer.Transactions, customer.AvgBalance))
    p_fused = float(lr.predict_proba(np.array([[p_hard, p_soft, change_score]]))[:,1][0])
    return {
        "P_Hard": round(p_hard,6),
        "P_Soft": round(p_soft,6),
        "Change_Score": round(change_score,6),
        "P_Fused": round(p_fused,6),
        "Threshold": THRESH,
        "Churn_Predicted": int(p_fused >= THRESH)
    }

@app.post("/batch_score")
async def batch_score(file: UploadFile = File(...)):
    """
    Upload a CSV with columns:
    CustomerID, Tenure, Transactions, AvgBalance, LoanHistory, CreditCardUsage, DefaultHistory
    Returns JSON rows with P_Hard/P_Soft/Change_Score/P_Fused/Churn_Predicted.
    """
    text = (await file.read()).decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(text))
    scored = score_dataframe(df)
    return {
        "threshold": THRESH,
        "rows": scored.to_dict(orient="records"),
        "count": int(len(scored))
    }
