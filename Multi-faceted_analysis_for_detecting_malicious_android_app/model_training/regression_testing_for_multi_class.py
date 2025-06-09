import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, log_loss, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler, label_binarize

# ---------- Load Dataset ----------
dataset = pd.read_csv('../androZoo_dataset_analysis/CICDataset_MalDroid_2020.csv')  # Replace with your dataset path
dataset.replace('?', np.nan, inplace=True)
dataset.dropna(inplace=True)
dataset = dataset.apply(pd.to_numeric, errors='ignore')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.astype(int)

print("Dataset shape:", X.shape)

# Adjust class labels if they start from 1 to 5 -> to zero-based for sklearn
y -= 1  # Convert classes 1-5 to 0-4

# ---------- Encode Target for ROC ----------
n_classes = len(np.unique(y))
y_bin = label_binarize(y, classes=np.arange(n_classes))

# ---------- Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

# ---------- Standardize Features ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- Base Logistic Regression ----------
model = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)

# ---------- Evaluation for Base Model ----------
base_acc = accuracy_score(y_test, y_pred)
base_loss = log_loss(y_test_bin, y_probs)
base_f1 = f1_score(y_test, y_pred, average='weighted')
base_precision = precision_score(y_test, y_pred, average='weighted')
base_recall = recall_score(y_test, y_pred, average='weighted')

# ---------- Grid Search ----------
param_grid = {
    'C': [1, 10],
    'solver': ['lbfgs', 'newton-cg'],
    'max_iter': [3000, 2000]
}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=3, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

print("\nBest Params from GridSearchCV:", grid.best_params_)
best_model = grid.best_estimator_
grid_preds = best_model.predict(X_test)
grid_probs = best_model.predict_proba(X_test)

# ---------- Evaluation for GridSearch Model ----------
grid_acc = accuracy_score(y_test, grid_preds)
grid_loss = log_loss(y_test_bin, grid_probs)
grid_f1 = f1_score(y_test, grid_preds, average='weighted')
grid_precision = precision_score(y_test, grid_preds, average='weighted')
grid_recall = recall_score(y_test, grid_preds, average='weighted')

# ---------- Print Results Summary ----------
print("\n================== Model Evaluation Summary ==================")
print(f"{'Metric':<15}{'Base Logistic':<20}{'GridSearch Logistic'}")
print(f"{'-'*55}")
print(f"{'Accuracy':<15}{base_acc:.4f}{'':<10}{grid_acc:.4f}")
print(f"{'Log Loss':<15}{base_loss:.4f}{'':<10}{grid_loss:.4f}")
print(f"{'F1 Score':<15}{base_f1:.4f}{'':<10}{grid_f1:.4f}")
print(f"{'Precision':<15}{base_precision:.4f}{'':<10}{grid_precision:.4f}")
print(f"{'Recall':<15}{base_recall:.4f}{'':<10}{grid_recall:.4f}")
print("==============================================================")
