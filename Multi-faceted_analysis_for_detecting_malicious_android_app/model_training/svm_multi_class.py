import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize

# Step 1: Load and preprocess dataset
dataset = pd.read_csv('../androZoo_dataset_analysis/reduced_CICDataset_MalDroid_2020.csv').iloc[:, 1:]

dataset.replace('?', np.nan, inplace=True)
dataset.dropna(inplace=True)
dataset = dataset.apply(pd.to_numeric, errors='ignore')

X = dataset.drop(columns=['Class']).values
y = dataset['Class'].values.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 2: Built-in multi-class SVM
print("\n==============================")
print("Starting Built-in Multi-class SVM Training & Evaluation")
print("==============================")

builtin_svm = SVC(kernel='linear', probability=True, random_state=42)
builtin_svm.fit(X_train, y_train)
builtin_preds = builtin_svm.predict(X_test)
builtin_probs = builtin_svm.predict_proba(X_test)  # shape (n_samples, n_classes)

builtin_accuracy = accuracy_score(y_test, builtin_preds)
builtin_precision = precision_score(y_test, builtin_preds, average='macro', zero_division=0)
builtin_recall = recall_score(y_test, builtin_preds, average='macro', zero_division=0)
builtin_f1 = f1_score(y_test, builtin_preds, average='macro', zero_division=0)

# Binarize labels for multi-class AUC
y_test_bin = label_binarize(y_test, classes=[1, 2, 3, 4, 5])
builtin_auc = roc_auc_score(y_test_bin, builtin_probs, average='macro', multi_class='ovr')

# Print results
print("\n===== Built-in Multi-class SVM Results =====")
print("Accuracy: {:.2f}%".format(builtin_accuracy * 100))
print("Macro AUC Score:", builtin_auc)
print("Macro Precision: {:.4f}".format(builtin_precision))
print("Macro Recall: {:.4f}".format(builtin_recall))
print("Macro F1 Score: {:.4f}".format(builtin_f1))
print("==============================")
print("Completed Built-in Multi-class SVM")
print("==============================")
