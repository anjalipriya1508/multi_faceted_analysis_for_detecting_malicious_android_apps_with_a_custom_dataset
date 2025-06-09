import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, 
    precision_score, recall_score, f1_score
)

# Step 1: Load and preprocess dataset
#dataset = pd.read_csv('../androZoo_dataset_analysis/TUANDROMD.csv').iloc[:, 1:]
dataset = pd.read_csv('../androZoo_dataset_analysis/synthetic_static_dynamic_network_dataset.csv').iloc[:, 1:]
#dataset = pd.read_csv('../androZoo_dataset_analysis/drebin.csv').iloc[:, 1:]

dataset.replace('?', np.nan, inplace=True)
dataset.dropna(inplace=True)
dataset = dataset.apply(pd.to_numeric, errors='ignore')

# Label encoding
#dataset['Label'] = dataset['Label'].map({'goodware': 0, 'malware': 1})
#dataset['class'] = dataset['class'].map({'B': 0, 'S': 1})  # Use for Drebin-style

# Split features and labels
X = dataset.drop(columns=['label']).values
y = dataset['label'].values.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Built-in SVM (linear)
builtin_svm = SVC(kernel='linear', probability=True, random_state=42)
builtin_svm.fit(X_train, y_train)
builtin_preds = builtin_svm.predict(X_test)
builtin_probs = builtin_svm.predict_proba(X_test)[:, 1]

builtin_accuracy = accuracy_score(y_test, builtin_preds)
builtin_auc = roc_auc_score(y_test, builtin_probs)
builtin_precision = precision_score(y_test, builtin_preds, zero_division=0)
builtin_recall = recall_score(y_test, builtin_preds, zero_division=0)
builtin_f1 = f1_score(y_test, builtin_preds, zero_division=0)

print("\n===== Built-in SVM =====")
print("Accuracy: {:.2f}%".format(builtin_accuracy * 100))
print("AUC Score:", builtin_auc)
print("Precision: {:.4f}".format(builtin_precision))
print("Recall: {:.4f}".format(builtin_recall))
print("F1 Score: {:.4f}".format(builtin_f1))

# Step 3: GridSearchCV on SVM
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("\n===== Grid Search Results: All Parameter Combinations =====")
cv_results = pd.DataFrame(grid_search.cv_results_)
params_scores = cv_results[['params', 'mean_test_score']]
for idx, row in params_scores.iterrows():
    print(f"Params: {row['params']}, Accuracy: {row['mean_test_score']:.4f}")

print("\n===== Grid Search Results =====")
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

grid_model = grid_search.best_estimator_
grid_preds = grid_model.predict(X_test)
grid_probs = grid_model.predict_proba(X_test)[:, 1]

grid_accuracy = accuracy_score(y_test, grid_preds)
grid_auc = roc_auc_score(y_test, grid_probs)
grid_precision = precision_score(y_test, grid_preds, zero_division=0)
grid_recall = recall_score(y_test, grid_preds, zero_division=0)
grid_f1 = f1_score(y_test, grid_preds, zero_division=0)

print("\n===== Best GridSearchCV SVM Evaluation =====")
print("Accuracy: {:.2f}%".format(grid_accuracy * 100))
print("AUC Score:", grid_auc)
print("Precision: {:.4f}".format(grid_precision))
print("Recall: {:.4f}".format(grid_recall))
print("F1 Score: {:.4f}".format(grid_f1))

# Step 4: ROC Curves
fpr_builtin, tpr_builtin, _ = roc_curve(y_test, builtin_probs)
fpr_grid, tpr_grid, _ = roc_curve(y_test, grid_probs)

def get_tpr_fpr_at_threshold(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    return TPR, FPR

tpr_b, fpr_b = get_tpr_fpr_at_threshold(y_test, builtin_probs)
tpr_g, fpr_g = get_tpr_fpr_at_threshold(y_test, grid_probs)

# Step 5: Plot ROC curves
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(fpr_builtin, tpr_builtin, label=f'AUC = {builtin_auc:.4f}', color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Built-in SVM ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(fpr_grid, tpr_grid, label=f'AUC = {grid_auc:.4f}', color='red')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('GridSearchCV SVM ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Final Summary Table
print("\n===== Summary (Accuracy %, Precision, Recall, F1, FPR) =====")
print(f"Built-in SVM     => Accuracy: {builtin_accuracy*100:.2f}%, Precision: {builtin_precision:.4f}, Recall: {builtin_recall:.4f}, F1 Score: {builtin_f1:.4f}, FPR: {fpr_b:.4f}")
print(f"GridSearchCV SVM => Accuracy: {grid_accuracy*100:.2f}%, Precision: {grid_precision:.4f}, Recall: {grid_recall:.4f}, F1 Score: {grid_f1:.4f}, FPR: {fpr_g:.4f}")
