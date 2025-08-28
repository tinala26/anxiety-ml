# Anxiety ML – GAD_T Prediction

End-to-end machine learning pipeline to predict **GAD_T (Generalized Anxiety Disorder threshold)** from survey data.

## What this project includes
- **Data preprocessing:** missing-value handling, categorical encoding, feature engineering (SWL, SPIN, Degree, etc.).
- **Models:**
  - Custom **LTCN Classifier** 
  - **Logistic Regression** (baseline)
  - **XGBoost**
- **Evaluation:**
  - Stratified K-Fold cross-validation
  - **Cohen’s Kappa**
  - Confusion matrices
  - t-SNE visualization

## How to run
```bash
pip install -r requirements.txt
python anxiety-ml.py
# anxiety-ml
# anxiety-ml
# anxiety-ml
# anxiety-ml
# anxiety-ml
# anxiety-ml
