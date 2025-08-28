# Libraries

import warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, cohen_kappa_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline        # we use imblearn Pipeline so SMOTE can sit inside
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from sklearn import linear_model

#Load 
csv_path = "anxiety-2.csv"         
data = pd.read_csv(csv_path, encoding="utf-8")
# cleaning 

for c in ['highestleague', 'S. No.', 'accept', 'Birthplace_ISO3', 'Residence_ISO3']:
    if c in data.columns:
        data = data.drop(columns=c) # drop columns 

# fill missing data
if 'GADE' in data:  data['GADE']  = data['GADE'].fillna(data['GADE'].mode()[0])
if 'Hours' in data: data['Hours'] = data['Hours'].fillna(data['Hours'].median())
if 'streams' in data: data['streams'] = data['streams'].fillna(data['streams'].median())
for c in ['Reference','Work']:
    if c in data.columns:
        data[c] = data[c].fillna(data[c].mode()[0])
if 'Narcissism' in data:
    data['Narcissism'] = data['Narcissism'].fillna(data['Narcissism'].median()).astype(int)
# Degree-ordinal transform 
if 'Degree' in data:
    data['Degree'] = (data['Degree'].astype(str).str.replace('�','').str.strip())
    data['Degree'] = data['Degree'].fillna(data['Degree'].mode()[0])
    data['Degree'] = data['Degree'].map({
        'High school diploma (or equivalent)': 0,
        'Bachelor(or equivalent)': 1,
        'Master(or equivalent)': 2,
        'Ph.D., Psy. D., MD (or equivalent)': 3
    }).fillna(0).astype(int)

# Gade-Target 
if 'GADE' in data:
    data['GADE'] = data['GADE'].map({
        'Not difficult at all': 0,
        'Somewhat difficult': 1,
        'Very difficult': 2,
        'Extremely difficult': 3
    }).fillna(0).astype(int)

# clean redudant words 
if 'League' in data:
    s = data['League'].astype(str).str.strip().str.lower().replace({'-': np.nan, 'none': np.nan})
    s = s.fillna('unranked')
    league = np.where(
        s.str.startswith(('gold','silv','plat','plati','dia','bron','mast','chall','unrank')),
        s, 'other'
    )
    # normalize names
    data['League'] = pd.Series(league).replace({
        'gold':'gold', 'silv':'silver', 'plat':'platinum', 'plati':'platinum',
        'dia':'diamond', 'bron':'bronze', 'mast':'master', 'chall':'challenger',
        'unrank':'unranked'
    })




# Timestamp 
if 'Timestamp' in data:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Year'] = data['Timestamp'].dt.year
    data = data.drop(columns=['Timestamp'])

# Make target binary 
data['GAD_T'] = (data['GAD_T'] >= 10).astype(int)

# EDA
df = data.copy()

if 'Gender' in df.columns:
    plt.figure()
    sns.barplot(x="Gender", y="GAD_T", data=df, ci=None)
    plt.title("Average GAD_T by Gender")
    plt.tight_layout(); plt.savefig("plot_gender_vs_target.png"); plt.close()

if 'Hours' in df.columns:
    plt.figure()
    sns.heatmap(df[['Hours','GAD_T']].corr(), annot=True, vmin=-1, vmax=1, cmap="coolwarm")
    plt.title("Correlation: Hours vs GAD_T")
    plt.tight_layout(); plt.savefig("plot_hours_corr.png"); plt.close()


# 
# drop GAD_ columns to avoid leakage WITH the target 
cat_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
leak_cols = [c for c in df.columns if c.startswith('GAD') and c != 'GAD_T']

X = df.drop(columns=leak_cols + ['GAD_T']).to_numpy()
y = df['GAD_T'].to_numpy()

# Train/test split ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

class LTCN:
    def __init__(self, T=10, phi=0.8, function="sigmoid", alpha=1e-2):
        self.T = T; self.phi = phi; self.function = function; self.alpha = alpha
        self.W1 = None; self.model = None

    def fit(self, X, y):
        Y = to_categorical(y, num_classes=len(np.unique(y)))
        if self.W1 is None:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            W = Vt.T
            self.W1 = W / np.max(np.abs(W))
        H = self._reason(X)
        self.model = linear_model.Ridge(alpha=self.alpha)
        self.model.fit(H, Y)
        return self

    def predict(self, X):
        H = self._reason(X)
        Yp = self.model.predict(H)
        return np.argmax(Yp, axis=1)

    def _reason(self, A):
        A0 = A; H = A0
        for _ in range(self.T):
            Z = A @ self.W1
            A = self.phi * (1 / (1 + np.exp(-Z)) if self.function == "sigmoid" else np.tanh(Z)) + (1 - self.phi) * A0
            H = np.concatenate((H, A), axis=1)
        return H
# Pipeline for imbalance 
log_pipe = Pipeline([
    ("sc", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

xgb_pipe = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    ))
])

ltcn_pipe = Pipeline([
    ("sc", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("clf", LTCN(T=10, phi=0.8, function="sigmoid", alpha=1e-2))
])




cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # Cross-validation

print("\nCv:")
for name, model in [("LTCN", ltcn_pipe), ("LogReg", log_pipe), ("XGB", xgb_pipe)]:
    kappas = []
    for tr, va in cv.split(X_train, y_train):
        X_tr, X_va = X_train[tr], X_train[va]
        y_tr, y_va = y_train[tr], y_train[va]
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_va)
        kappas.append(cohen_kappa_score(y_va, y_hat))
    print(f"  {name} folds:", [round(k,3) for k in kappas])
    print(f"  {name} mean:  {round(float(np.mean(kappas)), 4)}")


# Final test

print("\ntest Cohen's Kap:")
for name, model in [("LTCN", ltcn_pipe), ("LogReg", log_pipe), ("XGB", xgb_pipe)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}: {round(cohen_kappa_score(y_test, y_pred), 4)}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig(f"cm_{name.lower()}.png")
    plt.close()

print("\nSaved plots: ltcn.png, logreg.png, xgb.png")
