# Breast Cancer Diagnosis - Classification Models & Streamlit App

**Student:** Suraj Kumar | **Student ID:** 2026AJ05451
**Course:** AIMLZG565 Machine Learning | M.Tech. (AI & ML), BITS Pilani WILP | Assignment II

## a. Problem statement

Early and accurate classification of a breast mass as **benign** or **malignant** supports faster clinical
follow-up. The goal of this assignment is to build and compare five supervised classification models on a
public tumour-measurement dataset, evaluate them with six metrics (Accuracy, AUC, Precision, Recall, F1, MCC),
and expose the trained models through an interactive Streamlit web application deployed on Streamlit Community
Cloud. Malignant is treated as the positive class, so Recall measures how many cancers are caught and Precision
measures how many alarms are real.

## b. Dataset description

| Property | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic), WDBC |
| Source | UCI Machine Learning Repository / Kaggle (`breast-cancer.csv`) |
| Instances | 569 (minimum required: 500) |
| Features | 30 numeric (minimum required: 12), after dropping the `id` column |
| Target | `diagnosis`: B = benign (357, 62.7%), M = malignant (212, 37.3%) |
| Missing values | None |
| Problem type | Binary classification |

The 30 features come from digitised images of fine-needle aspirates of breast masses. Ten cell-nucleus
characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry,
fractal dimension) are each summarised three ways: **mean**, **standard error (se)** and **worst** (mean of the
three largest values).

**Preprocessing and split**
- Dropped `id`; encoded B = 0 and M = 1.
- Stratified 80/20 train/test split with `random_state=42`: 455 training rows, 114 test rows (72 B, 42 M).
  The test split is saved as `test_data.csv`.
- `StandardScaler` is applied inside the scikit-learn Pipeline for Logistic Regression, KNN and SVM, which are
  scale-sensitive. Decision Tree and Gaussian Naive Bayes use raw features.
- Each fitted pipeline is stored in `model/` as a `.joblib` file, so the app never retrains.

## c. GitHub repository link

https://github.com/surajk99/breast-cancer-classifier_ML_assignment

**Live Streamlit app:** `<PASTE STREAMLIT COMMUNITY CLOUD URL HERE AFTER DEPLOYMENT>`

```
breast-cancer-classifier_ML_assignment/
|-- app.py                  Streamlit application
|-- train_models.py         Trains and saves all five models, writes test_data.csv
|-- requirements.txt
|-- README.md
|-- test_data.csv           Held-out test split (30 features + diagnosis)
|-- data/breast-cancer.csv  Original dataset
|-- model/                  *.joblib pipelines, label_map.json, metrics.csv
|-- 2026AJ05451_Assignment2.ipynb   Notebook (run on BITS Virtual Lab)
```

## d. Models used

All five models were trained on the same training split and evaluated on the same 114-row test split.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

Hyperparameters: Logistic Regression (`max_iter=1000`); Decision Tree (`max_depth=5`, `min_samples_leaf=3`);
KNN (`k=7`); Gaussian Naive Bayes (defaults); SVM (RBF kernel, `C=1.0`, `probability=True`).

Confusion-matrix counts on the test split (positive class = malignant):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 71 | 1 | 3 | 39 |
| Decision Tree | 70 | 2 | 9 | 33 |
| KNN | 71 | 1 | 4 | 38 |
| Naive Bayes | 72 | 0 | 7 | 35 |
| Support Vector Machine | 72 | 0 | 3 | 39 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a linear model: 96.5% accuracy and the highest AUC (0.9960), so it ranks malignant cases above benign ones almost perfectly. It misses 3 of 42 malignant tumours and raises 1 false alarm. The classes in this dataset are close to linearly separable once features are standardised. |
| Decision Tree | The weakest model on every metric (accuracy 0.9035, MCC 0.7926). It missed 9 of 42 malignant tumours (Recall 0.7857), which is the most clinically costly error here. A single depth-limited tree has high variance on 455 training rows and gives a coarse probability estimate, which also explains the lowest AUC (0.9471). Its advantage is interpretability. |
| KNN | Accurate (95.6%) and well balanced, with 1 false positive and 4 false negatives. It depends on feature scaling, since unscaled area values would dominate the distance, so the scaler in the pipeline matters. AUC (0.9825) is a little lower than the linear and kernel models because k=7 gives coarse probability estimates. |
| Naive Bayes | Perfect Precision (no false positives) but Recall of only 0.8333, with 7 malignant cases missed. Its AUC (0.9934) is very high, so the ranking is good but the default 0.5 threshold is too conservative. The independence assumption is violated here, because radius, perimeter and area are strongly correlated, which skews its probabilities. It is the fastest model to train. |
| Support Vector Machine | Best overall: highest Accuracy (0.9737), F1 (0.9630) and MCC (0.9442). It has no false positives and 3 false negatives, tied for fewest misses with Logistic Regression. The RBF kernel on standardised features fits the mildly non-linear boundary well. Its AUC (0.9947) is within 0.0013 of Logistic Regression. |

**Overall winner for this dataset: Support Vector Machine (RBF kernel).** It leads on Accuracy, F1 and MCC and
ties for the lowest false-negative count. Logistic Regression is a very close second and is simpler and more
interpretable.

*Caveat:* the test set has only 114 rows, so one prediction changes accuracy by about 0.9 percentage points. The
gap between SVM and Logistic Regression is a single sample, so the ranking of the top two is not statistically
firm. The Decision Tree gap is large enough to be reliable. Results come from one stratified split.

## Streamlit app features

1. **CSV upload** of the test data (must contain the 30 feature columns and `diagnosis`; labels B/M or 0/1).
   A bundled-sample option is available.
2. **Model selection dropdown** covering all five models.
3. **Evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC) for the chosen model.
4. **Confusion matrix and classification report**, plus an ROC curve and a table comparing all five models
   on the uploaded data.

The app loads the saved pipelines from `model/` and does not retrain on uploaded data.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py        # optional: regenerates model/ and test_data.csv
streamlit run app.py
```
