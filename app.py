"""Streamlit front-end: Breast Cancer (WDBC) classifier evaluation dashboard.

The app only LOADS pre-trained pipelines from ./model and scores whatever test CSV
the user uploads. Nothing is re-fitted on the uploaded data.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
CLASS_NAMES = ["Benign (B)", "Malignant (M)"]

st.set_page_config(page_title="Tumour Diagnosis - Model Evaluator", page_icon="🔬", layout="wide")


@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}
    with open(MODEL_DIR / "label_map.json") as fh:
        meta = json.load(fh)
    return models, meta


def encode_target(series, label_map):
    """Accept either B/M text labels or 0/1 integers."""
    s = series.copy()
    if s.dtype == object or str(s.dtype).startswith("str"):
        s = s.astype(str).str.strip().str.upper().map(label_map)
    return s


def score(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    out = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba) if y.nunique() > 1 else float("nan"),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return out, pred, proba


models, meta = load_artifacts()
FEATURES, TARGET, LABEL_MAP = meta["features"], meta["target"], meta["label_map"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("1. Upload test CSV", type=["csv"],
                                help="Must contain all 30 feature columns and the 'diagnosis' column.")
    use_sample = st.checkbox("Use bundled test_data.csv instead", value=uploaded is None)
    choice = st.selectbox("2. Choose a model", list(MODEL_FILES.keys()))
    st.caption("Models are pre-trained and loaded from /model. No retraining happens here.")

st.title("🔬 Breast Cancer Diagnosis - Model Evaluation Dashboard")
st.write("Evaluate five pre-trained classifiers on the Wisconsin Diagnostic Breast Cancer data. "
         "Malignant is treated as the positive class.")

# ---------------------------------------------------------------- data
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = f"Uploaded file: {uploaded.name}"
elif use_sample:
    df = pd.read_csv(ROOT / "test_data.csv")
    source = "Bundled test_data.csv"
else:
    st.info("Upload a test CSV in the sidebar, or tick the box to use the bundled sample.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The CSV is missing required columns: {missing}")
    st.stop()

df = df.dropna(subset=FEATURES + [TARGET])
y = encode_target(df[TARGET], LABEL_MAP)
if y.isna().any():
    st.error("Target column has unrecognised labels. Use B/M or 0/1.")
    st.stop()
y = y.astype(int)
X = df[FEATURES]

st.subheader("Dataset preview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Features", len(FEATURES))
c3.metric("Malignant share", f"{y.mean():.1%}")
st.caption(source)
st.dataframe(df.head(10), width='stretch')

# ---------------------------------------------------------------- selected model
st.subheader(f"Results: {choice}")
res, pred, proba = score(models[choice], X, y)
cols = st.columns(6)
for col, (k, v) in zip(cols, res.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="PuBu", cbar=False, ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{choice}")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), width='stretch')

if y.nunique() > 1:
    fpr, tpr, _ = roc_curve(y, proba)
    fig2, ax2 = plt.subplots(figsize=(5, 3.6))
    ax2.plot(fpr, tpr, label=f"{choice} (AUC = {res['AUC']:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", label="Chance")
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.set_title("ROC curve")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

# ---------------------------------------------------------------- all models
st.subheader("All models on this data")
table = pd.DataFrame(
    [{"ML Model Name": n, **score(m, X, y)[0]} for n, m in models.items()]
).set_index("ML Model Name").round(4)
st.dataframe(table.style.highlight_max(axis=0, color="#cfe8cf"), width='stretch')
best = table["MCC"].idxmax()
st.success(f"Highest MCC on this data: **{best}** ({table.loc[best, 'MCC']:.4f})")
