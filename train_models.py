"""Train five classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset.

Outputs
-------
model/<name>.joblib   : fitted sklearn Pipeline (scaler + classifier where needed)
model/label_map.json  : class label -> integer mapping and feature order
model/metrics.csv     : evaluation metrics on the held-out test split
test_data.csv         : held-out test split (features + diagnosis) for the app
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

raw = pd.read_csv(ROOT / "data" / "breast-cancer.csv").drop(columns=["id"])
TARGET = "diagnosis"
FEATURES = [c for c in raw.columns if c != TARGET]
label_map = {"B": 0, "M": 1}  # Malignant is the positive class

X = raw[FEATURES]
y = raw[TARGET].map(label_map)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

# Held-out test file used by the Streamlit app (original B/M labels retained)
test_df = X_test.copy()
test_df[TARGET] = y_test.map({v: k for k, v in label_map.items()})
test_df.to_csv(ROOT / "test_data.csv", index=False)

models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED))]),
    "KNN": Pipeline([
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7))]),
    "Naive Bayes": Pipeline([
        ("clf", GaussianNB())]),
    "Support Vector Machine": Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED))]),
}

rows = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, MODEL_DIR / fname)

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
with open(MODEL_DIR / "label_map.json", "w") as f:
    json.dump({"label_map": label_map, "features": FEATURES, "target": TARGET}, f, indent=2)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(y_train.value_counts().to_dict(), y_test.value_counts().to_dict())
print(metrics.to_string(index=False))
