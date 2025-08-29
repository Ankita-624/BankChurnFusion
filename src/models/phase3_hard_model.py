import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------
# CONFIG
# ----------------------------
DATA_IN = "data/bank_customers_processed.csv"
MODEL_REPORTS = "reports"
RANDOM_STATE = 42

os.makedirs(MODEL_REPORTS, exist_ok=True)

# ----------------------------
# 1) LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_IN)
print("Dataset shape:", df.shape)
print(df.head())

# ----------------------------
# 2) SELECT FEATURES / TARGET
# ----------------------------
FEATURES = [
    "z_Tenure", "z_Transactions", "z_AvgBalance",
    "LoanHistory", "CreditCardUsage", "DefaultHistory"
]
TARGET = "Churn"

X = df[FEATURES]
y = df[TARGET]

# ----------------------------
# 3) TRAIN/TEST SPLIT (stratified to keep class ratio)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
print("Class distribution (train):")
print(y_train.value_counts(normalize=True).rename({0:"No",1:"Yes"}))

# ----------------------------
# 4) CLASS WEIGHTS FOR IMBALANCE
# ----------------------------
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
print("Computed class weights:", class_weights)

# ----------------------------
# 5) TRAIN A STRONGER, STILL INTERPRETABLE TREE
# ----------------------------
dt = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=7,            # deeper to catch minority class
    min_samples_split=30,   # avoid tiny overfit splits
    min_samples_leaf=15,
    class_weight=class_weights,
    random_state=RANDOM_STATE
)
dt.fit(X_train, y_train)

# ----------------------------
# 6) EVALUATE
# ----------------------------
y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
print("\n✅ HARD MODEL PERFORMANCE")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {roc:.4f}")
print("\nClassification Report (threshold=0.50):")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0,1], yticklabels=[0,1], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Hard Model")
plt.savefig(f"{MODEL_REPORTS}/03_confusion_matrix.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ----------------------------
# 7) EXTRACT RULES
# ----------------------------
rules = export_text(dt, feature_names=FEATURES)
rules_path = os.path.join(MODEL_REPORTS, "decision_tree_rules.txt")
with open(rules_path, "w") as f:
    f.write(rules)
print(f"\n📄 Rules saved to: {rules_path}")

# ----------------------------
# 8) PLOT TREE
# ----------------------------
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(dt, feature_names=FEATURES, class_names=["No Churn","Churn"],
          filled=True, rounded=True, fontsize=8, ax=ax)
plt.savefig(f"{MODEL_REPORTS}/04_decision_tree.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ----------------------------
# 9) FEATURE IMPORTANCE
# ----------------------------
importances = pd.Series(dt.feature_importances_, index=FEATURES).sort_values(ascending=False)
fig, ax = plt.subplots()
importances.plot(kind="bar", ax=ax)
ax.set_title("Feature Importance - Hard Model")
ax.set_ylabel("Importance")
plt.tight_layout()
plt.savefig(f"{MODEL_REPORTS}/05_feature_importance.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("\nTop Features:")
print(importances)

# ----------------------------
# 10) SAVE HARD PROBS FOR ALL CUSTOMERS
# ----------------------------
df["P_Hard"] = dt.predict_proba(X)[:, 1]
out_path = "data/churn_hard_scores.csv"
df.to_csv(out_path, index=False)
print(f"\n✅ Hard churn probabilities saved to: {out_path}")
