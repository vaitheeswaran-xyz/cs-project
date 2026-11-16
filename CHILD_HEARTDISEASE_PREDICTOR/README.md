# Child Heart Disease Predictor

Predicting **congenital heart disease (CHD) in children** from features extracted from 12-lead ECG recordings, using an **XGBoost** classifier.

📁 **Project files (Google Drive):** https://drive.google.com/drive/u/1/folders/1uVEawbejLCwjnBfFLWyuChALuzWm0yJK

| File | Description |
|------|-------------|
| `Xboost.ipynb` | Training, feature selection, threshold tuning and evaluation notebook |
| `ecg_features.csv` (in Drive, `ML_MODEL/`) | Pre-extracted ECG feature table used as input |
| `trained_model.pkl` (generated) | Saved XGBoost model produced by the notebook |

---

## 1. What does the notebook do?

`Xboost.ipynb` is a binary classifier. Given the ECG-derived measurements of a child's recording, it predicts:

- `0`: no congenital heart disease
- `1`: congenital heart disease (CHD)

It works on a tabular dataset, `ecg_features.csv`, with **1,784 recordings x 221 columns**. The columns are a `Record` ID (for example `P00001_E01`), a `Label`, and numeric ECG features. The record naming matches the public pediatric ECG database listed in the references.

The feature groups are:

- **Rhythm and rate:** RR interval and heart rate (mean, median, std, min, max).
- **ECG intervals:** PR, QRS, QT and QTc.
- **Wave morphology:** P, Q, R, S and T amplitudes, ratios, areas, energies, slopes and durations, plus ST level.
- **Heart-rate variability (HRV):** time-domain (CVNN, pNN50, ...), frequency-domain, Poincaré (SD1/SD2, ...), entropy (ApEn, SampEn, ...), and fractal/DFA/MFDFA measures.

> The feature extraction step (raw ECG → `ecg_features.csv`) is not part of this notebook. The notebook starts from the finished CSV. It installs `wfdb` because that is the usual library for reading PhysioNet ECG signals.

## 2. How does it work?

```
ecg_features.csv
      │
      ▼
1. Clean data ─ drop Label/Record, keep numeric columns, drop all-NaN columns,
      │         replace ±inf with NaN, fill NaN with column median      → 211 features
      ▼
2. Stratified 80/20 train/test split (random_state=42)                  → 1427 train / 357 test
      ▼
3. Class balance check (792 non-CHD vs 635 CHD in train, ratio ≈ 1.25)
      ▼
4. Feature selection: RFECV + XGBoost, 5-fold CV, scored on F1          → 111 features kept
      ▼
5. Train XGBClassifier (300 trees, depth 6, lr 0.05, subsample 0.8, colsample 0.8)
      ▼
6. Predict probabilities, then apply a custom decision threshold (0.26)
      ▼
7. Evaluate: classification report, confusion matrix, ROC and PR curves,
   threshold sweep, feature importance
      ▼
8. Save model with joblib → trained_model.pkl
```

Key design choices:

- **Median imputation** keeps all rows. Many HRV features are `NaN` for short recordings.
- **RFECV** is fitted on the training set only, so the test set doesn't influence feature selection. It removes 10 features per step and uses cross-validated F1 to pick the best subset size.
- **Threshold tuning:** the default 0.5 cutoff gave low recall (about 0.50). In a screening setting, missing a sick child is worse than a false alarm. The threshold was lowered to **0.26** to favour recall.

## 3. Results

Evaluated on the held-out test set (357 recordings: 198 non-CHD, 159 CHD) at threshold **0.26**:

| Metric | Score |
|--------|-------|
| Accuracy | **0.683** |
| Precision (CHD) | 0.614 |
| Recall / Sensitivity (CHD) | **0.780** |
| F1 (CHD) | 0.687 |
| ROC AUC | **0.756** |

**Confusion matrix**

|                 | Predicted non-CHD | Predicted CHD |
|-----------------|:-----------------:|:-------------:|
| **Actual non-CHD** | 120 (TN) | 78 (FP) |
| **Actual CHD**     | 35 (FN)  | 124 (TP) |

**Threshold trade-off (selected rows):**

| Threshold | Precision | Recall | F1 |
|:---------:|:---------:|:------:|:--:|
| 0.10 | 0.519 | 0.943 | 0.670 |
| 0.20 | 0.574 | 0.830 | 0.679 |
| 0.25 | 0.615 | 0.792 | 0.692 |
| 0.30 | 0.619 | 0.736 | 0.672 |
| 0.50 | 0.650 | 0.503 | 0.567 |
| 0.90 | 0.902 | 0.233 | 0.370 |

The best F1 from the precision-recall curve is about **0.701** at a threshold of about **0.243**.

**Most important features** include `QTc_Median`, `HR_STD`, `Q_Amp_STD`, `RS_Ratio_STD`, `QT_Median`, `HRV_PIP`, `R_Slope_Mean`, `HRV_pNN20` and `HR_Min`. Repolarisation timing (QT/QTc), heart-rate variability and Q/R/S wave shape contribute the most.

## 4. Outcomes and takeaways

- ECG-derived features carry real signal for CHD, with an AUC of about 0.76, well above random (0.5).
- At the chosen threshold the model catches about **78%** of CHD cases. The cost is about **39%** of healthy children flagged (78 of 198), so it is better suited as a **screening aid** than a diagnostic tool.
- Accuracy of about 68% is **not clinically sufficient** on its own. It should never replace a cardiologist or an echocardiogram.

**Limitations and possible improvements**

- The threshold (0.26) was chosen by looking at test-set results. This is slightly optimistic. A cleaner approach is to tune it on a validation set or with cross-validation, and keep the test set untouched.
- Only one train/test split was used. Repeated or stratified k-fold CV would give more reliable estimates.
- Splitting is at the recording level. Several recordings from the same patient (`P00032_E01`, `P00032_E02`) may land in both train and test, which could inflate scores. Use a **group-aware split by patient ID**.
- Try hyperparameter search (Optuna or `RandomizedSearchCV`), probability calibration, and other models (LightGBM, Random Forest, a 1D-CNN on raw ECG).
- Add SHAP explanations for per-patient interpretability.

## 5. How to run it

**Option A: Google Colab (recommended)**

1. Download or open the Drive folder linked above and put `ecg_features.csv` in `MyDrive/ML_MODEL/`.
2. Open `Xboost.ipynb` in Colab and choose **Runtime → Run all**.
3. Approve the Drive mount when prompted.

**Option B: Locally**

```bash
pip install wfdb scipy scikit-learn pandas numpy matplotlib joblib xgboost tqdm
jupyter notebook Xboost.ipynb
```

For a local run:

- Remove `from google.colab import drive` and `drive.mount(...)`.
- Change `CSV_PATH` to wherever you saved `ecg_features.csv`.

**Using the saved model**

```python
import joblib
model = joblib.load("trained_model.pkl")
# X_new must contain the same 111 selected features, in the same order
prob = model.predict_proba(X_new)[:, 1]
pred = (prob >= 0.26).astype(int)
```

## 6. References

1. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* Proceedings of the 22nd ACM SIGKDD. https://arxiv.org/abs/1603.02754
2. Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). *Gene Selection for Cancer Classification using Support Vector Machines* (Recursive Feature Elimination). Machine Learning, 46, 389-422.
3. Li, J., et al. *ZZU pECGdb: A Pediatric ECG Database*. PhysioNet. https://physionet.org/ (search "pediatric ECG"). Check the exact dataset page for the official citation.
4. Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation, 101(23), e215-e220.
5. Makowski, D., et al. (2021). *NeuroKit2: A Python toolbox for neurophysiological signal processing.* Behavior Research Methods, 53, 1689-1696.
6. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825-2830.
7. Task Force of the ESC and NASPE (1996). *Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use.* Circulation, 93(5), 1043-1065.
8. Saito, T., & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLoS ONE, 10(3).

## Disclaimer

This is an educational/research project. It is **not a medical device** and must not be used for real clinical decisions.
