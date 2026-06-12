# MigraineMap
### Multi-Class Migraine Subtype Classification with ICHD-3 Validated Explainability and Family Inference Engine

> *Can a machine learning model trained on clinical symptom profiles correctly classify seven ICHD-3 defined migraine subtypes — and can the features it identifies as most important be predicted from established clinical criteria before the model runs?*

---

## Overview

MigraineMap is a machine learning pipeline for classifying seven clinically-defined migraine subtypes from patient symptom profiles. It trains and compares four classical ML models on 23 ICHD-3 clinical features, applies SHAP explainability with pre-specified biological predictions validated against the International Classification of Headache Disorders (ICHD-3), and includes a custom family inference engine that applies the trained classifier to manually constructed symptom profiles with Deficit Penetrance Factor (DPF) sensitivity analysis and pairwise concordance scoring.

**Personal motivation:** My mother, sister, and grandmother all have migraines. I have an undiagnosed atypical presentation. This project was built to understand the clinical and symptomatic architecture of migraine subtypes across four generations of my family.

---

## The Biology

Migraine is not a single disease. The ICHD-3 international classification defines seven distinct subtypes with different symptom signatures, different neurological mechanisms, and different treatment responses:

| Subtype | Key Distinguishing Features |
|---|---|
| Typical aura with migraine | Visual/sensory aura followed by headache |
| Migraine without aura | Recurrent headache without preceding neurological symptoms |
| Familial hemiplegic migraine | Motor aura with family history — near-Mendelian inheritance |
| Sporadic hemiplegic migraine | Motor aura without family history |
| Basilar-type aura | Brainstem-origin symptoms: vertigo, tinnitus, diplopia |
| Typical aura without migraine | Aura without subsequent headache |
| Other | Presentations not fitting standard ICHD-3 classifications |

Accurate subtype classification may assist clinical decision support and subtype differentiation, particularly for migraine variants with overlapping symptom profiles.
---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle — `ranzeet013/migraine-dataset` |
| Patients | 400 |
| Features | 23 ICHD-3 clinical features |
| Target | 7 migraine subtypes |
| Missing values | None |

**Class distribution:**

| Subtype | Count | Note |
|---|---|---|
| Typical aura with migraine | 247 | Majority class |
| Migraine without aura | 60 | — |
| Familial hemiplegic migraine | 24 | — |
| Typical aura without migraine | 20 | — |
| Basilar-type aura | 18 | — |
| Other | 17 | — |
| Sporadic hemiplegic migraine | 14 | Minority class |

**Important limitation:** With 80 test samples across 7 classes, minority class metrics (Basilar-type n=4, Sporadic hemiplegic n=3 in test set) are statistically unreliable and should not be interpreted as standalone performance claims. Macro-averaged cross-validation F1 across 100 RepeatedStratifiedKFold evaluations provides the only reliable generalisation estimate.

---

## Pipeline

```
Kaggle Dataset (400 patients, 23 features, 7 subtypes)
          │
          ▼
LabelEncoder → Stratified 80/20 split (random_state=42)
          │
          ▼
Four models trained with class_weight='balanced'
RF · XGBoost · Logistic Regression · SVM
          │
          ▼
RepeatedStratifiedKFold (5×20 = 100 evaluations)
Pipeline-based CV for LR/SVM — scaler inside each fold
          │
          ▼
Overfitting check — Train vs Test Macro F1 gap
          │
          ▼
SHAP TreeExplainer → Global + per-subtype feature importance
Pre-specified ICHD-3 predictions validated against SHAP output
          │
          ▼
Family Inference Engine
predict_proba() → DPF sensitivity analysis → pairwise concordance
```

---

## Models and Results

### Test Set Performance (80 samples)

| Model | Accuracy | Macro F1 | CV Mean F1 | CV Gap |
|---|---|---|---|---|
| Random Forest | 0.875 | 0.774 | 0.773 | 0.001 |
| XGBoost | 0.900 | 0.816 | 0.711 | 0.105 |
| Logistic Regression | 0.800 | 0.791 | 0.762 | 0.029 |
| SVM | 0.763 | 0.740 | 0.775 | -0.012 |

All models substantially above the majority class baseline (0.618).

XGBoost achieved the highest test-set performance but showed the largest train-to-cross-validation performance gap, suggesting mild overfitting relative to Random Forest.

**Model selection for family inference:** Random Forest selected over XGBoost despite lower test accuracy. CV gap of 0.001 vs XGBoost's 0.105 — Random Forest demonstrates the most stable generalisation across folds. For an application involving unseen family profiles, stability is preferred over marginal accuracy gains.

### Confusion Matrix (Random Forest)

The confusion matrix for Random Forest on the 80-sample test set reveals the most clinically meaningful error pattern: **Basilar-type aura is most frequently confused with Familial hemiplegic migraine**. Both subtypes share strong family history signals and brainstem-origin symptoms, making them the hardest pair to distinguish from symptom profiles alone. This is consistent with clinical practice — even neurologists require detailed history and genetic testing to differentiate these subtypes reliably.

---

## SHAP Explainability

### Methodology

SHAP TreeExplainer was applied to the trained Random Forest model. **Three features were pre-specified as expected top features based on ICHD-3 criteria before running SHAP** — an independent prediction that was then confirmed by the model output. This pre-specification prevents post-hoc rationalisation and constitutes consistency with established ICHD-3 diagnostic criteria rather than a discovery claim.

### Global Feature Importance

| Rank | Feature | SHAP Value | Pre-specified? | ICHD-3 Basis |
|---|---|---|---|---|
| 1 | Visual aura | 0.044 | ✓ Yes | Definitional criterion for MwA — absence defines MwoA |
| 2 | DPF | 0.033 | ✓ Yes | Consistent with 40-60% heritability estimates for migraine |
| 3 | Age | 0.030 | No | Age of onset differs by subtype |
| 4 | Vertigo | 0.028 | ✓ Yes | Brainstem symptom — ICHD-3 criterion for Basilar-type |
| 5 | Intensity | 0.025 | No | Pain severity differs by subtype |

All three pre-specified features confirmed in top 4. The model independently recovered the same features that clinical diagnostic criteria prioritise.

### Per-Subtype Findings

| Subtype | Top Feature | Biological Consistency |
|---|---|---|
| Basilar-type aura | Vertigo | ✓ ICHD-3: brainstem-origin symptoms define this subtype |
| Familial hemiplegic migraine | DPF | ✓ Near-Mendelian inheritance — family history is primary discriminator |
| Migraine without aura | Visual (absence) | ✓ Defined by absence of aura |
| Typical aura with migraine | Visual | ✓ Defined by presence of visual aura |
| Sporadic hemiplegic migraine | DPF (low) | ✓ Distinguished from FHM by absence of family history |

---

## Family Inference Engine

### What It Is

A custom inference script applying the trained Random Forest classifier to manually constructed symptom profiles. This is a **computational demonstration tool**, not a clinical diagnostic instrument. All outputs should be interpreted as exploratory probability distributions, not diagnoses. Family profiles were manually constructed from self-reported symptoms and were not used during model training.

### DPF Sensitivity Analysis

For each family member, predictions are rerun with DPF (family history flag) set to zero. This feature ablation measures how much the family history signal — rather than the individual's own symptom pattern — drives the classification. Members whose primary classification changes significantly under ablation are flagged with a DPF warning.

### Four-Generation Family Analysis

| Member | Primary Classification | Confidence | DPF Flag |
|---|---|---|---|
| Grandmother (70) | Migraine without aura | 43.3% | No |
| Mother (47) | Migraine without aura | 38.6% | No |
| Sister (18) | Familial hemiplegic migraine | 41.0% | Yes — DPF driving classification |
| Anusha (20) | Other | 26.5% | Yes — symptom-only: MwoA (26.6%) |

**Pairwise concordance scores (Pearson correlation of probability distributions):**

| Pair | Concordance | Interpretation |
|---|---|---|
| Mother ↔ Grandmother | 0.761 | HIGH — similar symptom presentations |
| Sister ↔ Anusha | 0.470 | Moderate |
| Mother ↔ Anusha | 0.520 | Moderate |
| Sister ↔ Mother | 0.454 | Moderate |
| Grandmother ↔ Anusha | 0.138 | Low |
| Sister ↔ Grandmother | -0.026 | None — divergent symptom profiles |

**Observation:** High mother-grandmother concordance (0.761) alongside near-zero sister-grandmother concordance (-0.026) indicates differing symptom patterns across generations within the same family. This is consistent with known phenotypic heterogeneity in migraine — different family members can present with distinct subtypes despite shared familial risk.

**Disclaimer:** These results are outputs of a classifier trained on a 400-row Kaggle dataset applied to manually constructed symptom profiles of 4 individuals. They do not constitute genetic analysis, clinical diagnosis, or evidence of any biological mechanism. Consult a qualified neurologist for clinical evaluation.

---

## Limitations

1. **400 samples across 7 classes** — minority class metrics on the 80-sample test set are statistically unreliable. Macro-averaged RepeatedStratifiedKFold CV is the reliable performance estimate.

2. **Pre-curated Kaggle dataset** — not raw clinical records. Data provenance and collection methodology are not fully documented.

3. **Family module is a symptom-based inference engine** — not genetic testing. Predictions reflect symptom cluster similarity to training data, not genomic analysis.

4. **SHAP findings represent consistency with ICHD-3 criteria** — not novel biomarker discovery. The model recovers known clinical discriminators, which validates its learning but does not constitute new biological knowledge.

5. **EDA correlation heatmap measures linear relationships** — Random Forest and XGBoost are non-linear models. EDA identified strong linear separators subsequently corroborated by non-linear SHAP explainers, providing convergent evidence across both analytical approaches.

6. **Calibration of probability estimates not formally evaluated** — predict_proba() outputs are used in the family module but are not calibrated. Isotonic regression or Platt scaling would be required for reliable probability interpretation.

7. External validation on an independent clinical cohort was not performed.

---

## Figures

| Figure | Description |
|---|---|
| `class_distribution.png` | Class imbalance bar chart + symptom prevalence heatmap by subtype |
| `accuracy_f1.png` | Model comparison (Accuracy vs Macro F1) + per-class F1 heatmap |
| `confusion_matrix.png` | Random Forest confusion matrix on 80-sample test set |
| `global_shap.png` | Global SHAP feature importance across all classes |
| `all_subtypes.png` | Top 8 SHAP features per migraine subtype (7 panels) |

---

## Repository Structure

```
Migraine_Map/
├── Migraine_Map.ipynb           # Complete pipeline notebook
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── figures/
    ├── class_distribution.png
    ├── accuracy_f1.png
    ├── confusion_matrix.png
    ├── global_shap.png
    └── all_subtypes.png
```

---

## Technologies

| Library | Purpose |
|---|---|
| pandas, numpy | Data loading and manipulation |
| scikit-learn | RF, LR, SVM, preprocessing, CV, metrics |
| xgboost | Gradient boosting classifier |
| shap | TreeExplainer, global and per-class explainability |
| matplotlib, seaborn | Visualisation |
| kagglehub | Dataset download |
| scipy | Pearson concordance computation |

---

## Academic Context

This project uses the ICHD-3 (International Classification of Headache Disorders, 3rd edition) as the biological validation framework. Feature importance findings are evaluated against ICHD-3 diagnostic criteria rather than claimed as novel discoveries.

**Data source:** Kaggle dataset `ranzeet013/migraine-dataset`.

**Classification framework:** Headache Classification Committee of the International Headache Society (IHS). The International Classification of Headache Disorders, 3rd edition. Cephalalgia. 2018.

---

## Figures

All figures generated during analysis are available in the `/figures` directory.

---

*Built by Anusha Dalal — Computer Engineering, SPIT Mumbai (2024–2028)*
