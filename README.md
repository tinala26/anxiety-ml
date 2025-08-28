# Anxiety ML – GAD_T Prediction

End-to-end machine learning pipeline to predict **GAD_T (Generalized Anxiety Disorder threshold)** from survey data.

## What this project includes
- **Data preprocessing:** missing-value handling, categorical encoding, feature engineering (SWL, SPIN, Degree, etc.).
- **Models:**
  - **LTCN Classifier** 
  - **Logistic Regression** (baseline)
  - **XGBoost**
- **Evaluation:**
  - Stratified K-Fold cross-validation
  - **Cohen’s Kappa**
  - Confusion matrices
  - t-SNE visualization

