# Income Classification with a Gender-Fairness Audit Using Machine Learning

**Course:** B.Sc. (Hons.) Computer Science, Semester VI  
**Paper:** Machine Learning Applications  
**Institution:** Department of Computer Science, University of Delhi  
**Student:** Ekta Sachdeva  
**University Roll Number:** 24020570063  
**Problem Number:** 24  
**Domain:** Gender / Algorithmic Fairness  
**UN Sustainable Development Goals (SDGs):**  
- **SDG 5:** Gender Equality  
- **SDG 10:** Reduced Inequalities  

---

## 📌 Project Overview

Automated machine learning decision systems deployed in credit risk underwriting, automated employment screening, and public benefit entitlement scoring often inherit, institutionalize, and amplify historical societal inequalities present in empirical census data.

This project delivers an end-to-end, reproducible implementation predicting whether an individual's annual income exceeds **$50,000** using the **UCI Adult Census Income Dataset** (~48,842 records). Beyond raw predictive performance, we perform an in-depth **Gender Fairness Audit** across four canonical classification models:
1. **Logistic Regression** (L2 Regularized)
2. **Decision Tree Classifier** (Pruned / Complexity-bounded)
3. **Gaussian Naive Bayes**
4. **K-Nearest Neighbours** ($k$-NN)

We formulate and compute mathematical fairness metrics (**Demographic Parity Difference, Disparate Impact Ratio, Equal Opportunity Difference, Equalized Odds Difference, and Error Rate Disparities**). We demonstrate empirically why **Fairness Through Unawareness** (dropping `sex`) fails due to proxy encoding (e.g. `relationship: Husband/Wife`), implement **Kamiran & Calders Pre-processing Sample Re-weighting**, evaluate **Group Threshold Optimization**, and analyze the fundamental Pareto trade-offs between algorithmic fairness and predictive utility.

---

## 📁 Repository Structure

```text
du_ml_fairness_project/
│
├── Income_Classification_Gender_Fairness_Audit.ipynb  # Interactive Jupyter Notebook (Google Colab ready)
├── income_fairness_audit.py                          # Complete, self-contained modular Python script
├── README.md                                         # Comprehensive project documentation
└── figures/                                          # (Generated during run) High-resolution figures (300 DPI)
    ├── fig1_gender_distribution.png
    ├── fig2_income_distribution.png
    ├── fig3_income_by_gender.png
    ├── fig4_education_distribution.png
    ├── fig5_age_distribution.png
    ├── fig6_gender_income_percentages.png
    ├── fig7_correlation_matrix.png
    ├── fig8_roc_curves.png
    ├── fig9_confusion_matrices.png
    ├── fig10_fairness_audit_baseline.png
    └── fig11_fairness_utility_tradeoffs.png
```

---

## 🚀 How to Run the Project

### Option A: Running in Google Colab (Recommended for Academic Review)
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `Income_Classification_Gender_Fairness_Audit.ipynb`.
3. In the first cell, ensure the dependency installation is executed:
   ```bash
   !pip install ucimlrepo fairlearn scikit-learn pandas numpy matplotlib seaborn scipy
   ```
4. Click **Runtime** > **Run all** (`Ctrl + F9`). The notebook automatically fetches the dataset from the official UCI repository and executes all visualisations, cross-validations, fairness audits, and bias mitigation routines.

### Option B: Running the Python Script Locally
Ensure Python 3.8+ is installed on your machine.
1. Install dependencies:
   ```bash
   pip install ucimlrepo fairlearn scikit-learn pandas numpy matplotlib seaborn scipy
   ```
2. Run the script:
   ```bash
   python income_fairness_audit.py
   ```
3. All 11 figures are automatically rendered at publication-grade 300 DPI into the `figures/` directory, and an academic summary report is saved to `audit_academic_summary.txt`.

---

## 🔬 Experimental Methodology & Key Findings

### 1. Data Cleaning & Leakage Prevention
- **Missing Value Handling:** Replaced `'?'` strings with `np.nan` and applied mode/Unknown imputation.
- **Duplicate Removal:** Identified and dropped duplicate records.
- **Leakage Prevention:** Continuous scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) are strictly fit on the training partition ($80\%$) and transformed on the test partition ($20\%$) within scikit-learn `Pipeline` objects.
- **Stratified Partitioning:** Stratified on the joint interaction of `sex` and `income` ($S \times Y$) to preserve representation across all subgroups.

### 2. Machine Learning Benchmark (5-Fold Stratified CV)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV F1-Score (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **85.1%** | 0.738 | 0.589 | **0.655** | **0.905** | 0.654 ± 0.007 |
| **Decision Tree (Pruned)** | 84.8% | 0.760 | 0.536 | 0.628 | 0.887 | 0.627 ± 0.008 |
| **Gaussian Naive Bayes** | 80.2% | 0.563 | 0.652 | 0.604 | 0.865 | 0.603 ± 0.006 |
| **$k$-Nearest Neighbours** | 83.2% | 0.687 | 0.584 | 0.631 | 0.871 | 0.630 ± 0.009 |

*Note: Accuracy is deceptive due to 76:24 class imbalance. Logistic Regression is selected as the top benchmark model based on F1-Score and ROC-AUC.*

### 3. Baseline Gender Fairness Audit
- **Demographic Parity Gap:** Male approval rate ~22.6%, Female approval rate ~7.0% ($\Delta_{\text{DP}} \approx 0.156$).
- **Disparate Impact Ratio:** $DI \approx 0.31$, violating the US EEOC Four-Fifths (80%) Rule ($0.80$).
- **Error Rate Disparity (Disparate Mistreatment):** False Negative Rate for females is **$\approx 43.1\%$** vs. **$\approx 36.8\%$** for males. Qualified women are disproportionately misclassified into the low-income bracket.

### 4. Bias Mitigation Results (Pareto Analysis)
| Intervention Strategy | Accuracy | F1-Score | ROC-AUC | Disparate Impact ($DI$) | Demographic Parity Gap | Equal Opportunity Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline (Unmitigated)** | 85.1% | 0.655 | 0.905 | 0.310 | 0.156 | 0.063 |
| **2. Unawareness (Drop Gender)** | 84.9% | 0.651 | 0.903 | 0.321 | 0.152 | 0.061 |
| **3. Sample Re-weighting** | 83.9% | 0.642 | 0.901 | **0.725** | **0.068** | **0.018** |
| **4. Group Thresholding** | 84.1% | 0.648 | 0.905 | **0.814** | **0.045** | **0.009** |

---

## ⚖️ Sociotechnical & SDG Alignment
- **SDG 5 (Gender Equality):** Unmitigated automated systems penalize women by inflating False Negative Rates, perpetuating historical wage disparities under the guise of mathematical neutrality.
- **SDG 10 (Reduced Inequalities):** Pre-processing sample re-weighting successfully decouples algorithmic predictions from historical bias with minimal utility loss (<1.2% accuracy drop).
- **Theoretical Limitation:** Under Kleinberg's and Chouldechova's Impossibility Theorems, Demographic Parity and Equalized Odds cannot be simultaneously satisfied when baseline rates differ across demographic groups. Technical mitigation must therefore be paired with policy governance.

---
**Author:** Ekta Sachdeva (University Roll No: 24020570063)  
Department of Computer Science, University of Delhi
