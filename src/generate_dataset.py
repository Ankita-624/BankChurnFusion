import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers
N = 10000

# 1. CustomerID
customer_ids = [f"CUST{1000 + i}" for i in range(N)]

# 2. Age (18 to 70)
ages = np.random.randint(18, 70, N)

# 3. Gender (M/F)
genders = np.random.choice(['Male', 'Female'], N)

# 4. Income (₹2L to ₹25L per year)
incomes = np.random.randint(200000, 2500000, N)

# 5. Account Tenure (L) in years: 0 to 25
tenures = np.random.randint(0, 26, N)

# 6. Transactions per year (F): based on activity level
transactions = np.random.poisson(lam=100, size=N) + np.random.randint(0, 50, N)

# 7. Average Balance (M): proportional to income but with variation
avg_balances = np.round(np.random.normal(incomes * 0.3, 50000), 2)
avg_balances = np.clip(avg_balances, 0, None)  # No negatives

# 8. Loan History: 1 = Has active loans, 0 = No loans
loan_history = np.random.choice([0, 1], N, p=[0.6, 0.4])

# 9. Credit Card Usage: 1 = Uses credit card, 0 = No card
credit_card_usage = np.random.choice([0, 1], N, p=[0.5, 0.5])

# 10. Default History: 1 = Has defaulted before, 0 = Never defaulted
default_history = np.random.choice([0, 1], N, p=[0.85, 0.15])

# 11. Churn Label: We'll simulate churn using rules
# Logic: High churn probability if:
# - Low tenure (<3 years)
# - Low balance (<₹1L)
# - High defaults
# - Very low transactions (<30/year)
churn_probs = (
    (tenures < 3) * 0.35 +
    (avg_balances < 100000) * 0.3 +
    (transactions < 30) * 0.2 +
    (default_history == 1) * 0.15
)

# Add randomness to churn
churn = np.random.binomial(1, churn_probs.clip(0, 0.95))

# Create the DataFrame
df = pd.DataFrame({
    "CustomerID": customer_ids,
    "Age": ages,
    "Gender": genders,
    "AnnualIncome": incomes,
    "Tenure": tenures,
    "Transactions": transactions,
    "AvgBalance": avg_balances,
    "LoanHistory": loan_history,
    "CreditCardUsage": credit_card_usage,
    "DefaultHistory": default_history,
    "Churn": churn
})

# Create data directory if not exists
os.makedirs("data", exist_ok=True)

# Save CSV
file_path = "data/bank_customers.csv"
df.to_csv(file_path, index=False)

print(f"✅ Synthetic dataset generated successfully: {file_path}")
print(df.head())
