import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, log_loss, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt

# ---------- Step 1: Load and Preprocess Dataset ----------
dataset = pd.read_csv('../androZoo_dataset_analysis/TUANDROMD.csv').iloc[:, 1:]
#dataset = pd.read_csv('../androZoo_dataset_analysis/synthetic_static_dynamic_network_dataset.csv').iloc[:, 1:]
#dataset = pd.read_csv('../androZoo_dataset_analysis/drebin.csv').iloc[:, 1:]
dataset.replace('?', np.nan, inplace=True)
dataset.dropna(inplace=True)
dataset = dataset.apply(pd.to_numeric, errors='ignore')
dataset[dataset.columns[-1]] = dataset[dataset.columns[-1]].map({'goodware': 0, 'malware': 1})
#dataset[dataset.columns[-1]] = dataset[dataset.columns[-1]].map({'B': 0, 'S': 1})
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.astype(int)

print("Dataset size:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Manual Logistic Regression Utilities ----------

from scipy.special import expit

def sigmoid(z):
    return expit(z)

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def predict_prob(X, weights):
    return sigmoid(np.dot(X, weights))

def predict(X, weights, threshold=0.5):
    return (predict_prob(X, weights) >= threshold).astype(int)

def compute_loss(y_true, y_pred_prob, weights, l2_lambda=0.01):
    epsilon = 1e-15
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    l2_loss = (l2_lambda / 2) * np.sum(weights[1:] ** 2)
    return -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob)) + l2_loss

def standardize(X_train, X_val=None):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_std = (X_train - mean) / std
    if X_val is not None:
        X_val_std = (X_val - mean) / std
        return X_train_std, X_val_std
    return X_train_std

def fit_manual_logistic_regression(
    X_train, y_train, X_val=None, y_val=None,
    lr=0.05, epochs=2000, l2_lambda=0.01, batch_size=64,
    early_stopping=True, patience=100, decay=0.99, verbose=True
):
    if X_val is not None:
        X_train, X_val = standardize(X_train, X_val)
    else:
        X_train = standardize(X_train)

    X_train = add_bias(X_train)
    if X_val is not None:
        X_val = add_bias(X_val)

    n_samples, n_features = X_train.shape
    weights = np.random.normal(scale=0.01, size=n_features)

    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            xb = X_train[start:end]
            yb = y_train[start:end]

            preds = predict_prob(xb, weights)
            error = preds - yb
            grad = (np.dot(xb.T, error) / yb.size)
            grad[1:] += l2_lambda * weights[1:]
            weights -= lr * grad

        lr *= decay

        if X_val is not None and y_val is not None:
            val_preds = predict_prob(X_val, weights)
            val_loss = compute_loss(y_val, val_preds, weights, l2_lambda)
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}, LR={lr:.6f}")
            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        return weights
        else:
            train_preds = predict_prob(X_train, weights)
            train_loss = compute_loss(y_train, train_preds, weights, l2_lambda)
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, LR={lr:.6f}")

    return weights

def prepare_test(X_test, X_train_full):
    mean = X_train_full.mean(axis=0)
    std = X_train_full.std(axis=0) + 1e-8
    X_test_std = (X_test - mean) / std
    return add_bias(X_test_std)

# ---------- Validation Split ----------
X_train_full, X_val, y_train_full, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# ---------- Train Manual Logistic Regression ----------
manual_weights = fit_manual_logistic_regression(
    X_train_full, y_train_full, X_val, y_val,
    lr=0.05, epochs=2000, l2_lambda=0.01,
    batch_size=64, early_stopping=True, patience=150, decay=0.99, verbose=True
)

X_test_prepared = prepare_test(X_test, X_train_full)
manual_test_probs = sigmoid(np.dot(X_test_prepared, manual_weights))
manual_test_preds = (manual_test_probs >= 0.5).astype(int)
manual_accuracy = accuracy_score(y_test, manual_test_preds)
manual_loss = compute_loss(y_test, manual_test_probs, manual_weights, l2_lambda=0.01)
manual_precision = precision_score(y_test, manual_test_preds)
manual_recall = recall_score(y_test, manual_test_preds)
manual_f1 = f1_score(y_test, manual_test_preds)

# ---------- Built-in Logistic Regression ----------
builtin_model = LogisticRegression(penalty='l2', solver='liblinear', C=1, max_iter=500)
builtin_model.fit(X_train, y_train)
builtin_preds = builtin_model.predict(X_test)
builtin_probs = builtin_model.predict_proba(X_test)[:, 1]
builtin_accuracy = accuracy_score(y_test, builtin_preds)
builtin_loss = log_loss(y_test, builtin_probs)
builtin_precision = precision_score(y_test, builtin_preds)
builtin_recall = recall_score(y_test, builtin_preds)
builtin_f1 = f1_score(y_test, builtin_preds)

# ---------- Grid Search Logistic Regression ----------
param_grid = {
    'C': [0.1, 10, 15, 20, 25],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2'],
    'max_iter': [500, 1000]
}
grid_model = LogisticRegression()
grid_search = GridSearchCV(grid_model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

results = pd.DataFrame(grid_search.cv_results_)
results_to_display = results[['params', 'mean_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
print("\n===== All Grid Search Combinations (sorted by accuracy) =====")
for i, row in results_to_display.iterrows():
    print(f"Rank {int(row['rank_test_score'])}: Params = {row['params']}, Mean CV Accuracy = {row['mean_test_score']:.4f}")
print("\n===== Grid Search Results =====")
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
grid_preds = best_model.predict(X_test)
grid_probs = best_model.predict_proba(X_test)[:, 1]
grid_accuracy = accuracy_score(y_test, grid_preds)
grid_loss = log_loss(y_test, grid_probs)
grid_precision = precision_score(y_test, grid_preds)
grid_recall = recall_score(y_test, grid_preds)
grid_f1 = f1_score(y_test, grid_preds)

# ---------- Model Comparison ----------
print("\n===== Model Comparison (Accuracy, Loss) =====")
print(f"Manual Logistic Regression  - Accuracy: {manual_accuracy:.4f}, Loss: {manual_loss:.4f}")
print(f"Built-in Logistic Regression - Accuracy: {builtin_accuracy:.4f}, Loss: {builtin_loss:.4f}")
print(f"Grid Search Logistic Regression - Accuracy: {grid_accuracy:.4f}, Loss: {grid_loss:.4f}")

print("\n===== Accuracy, Precision, Recall, F1-Score Summary =====")
print("Manual Logistic Regression:")
print(f"Accuracy: {manual_accuracy:.4f},  Precision: {manual_precision:.4f}, Recall: {manual_recall:.4f}, F1-Score: {manual_f1:.4f}")
print("Built-in Logistic Regression:")
print(f" Accuracy: {builtin_accuracy:.4f}, Precision: {builtin_precision:.4f}, Recall: {builtin_recall:.4f}, F1-Score: {builtin_f1:.4f}")
print("Grid Search Logistic Regression:")
print(f" Accuracy: {grid_accuracy:.4f}, Precision: {grid_precision:.4f}, Recall: {grid_recall:.4f}, F1-Score: {grid_f1:.4f}")

# ---------- ROC Curve ----------
fpr_manual, tpr_manual, _ = roc_curve(y_test, manual_test_probs)
fpr_builtin, tpr_builtin, _ = roc_curve(y_test, builtin_probs)
fpr_grid, tpr_grid, _ = roc_curve(y_test, grid_probs)

roc_auc_manual = roc_auc_score(y_test, manual_test_probs)
roc_auc_builtin = roc_auc_score(y_test, builtin_probs)
roc_auc_grid = roc_auc_score(y_test, grid_probs)

def plot_roc_curve(fpr, tpr, roc_auc, accuracy, model_name, color):
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}\nAccuracy = {accuracy*100:.2f}%')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

plot_roc_curve(fpr_manual, tpr_manual, roc_auc_manual, manual_accuracy, "Manual Logistic Regression", 'blue')
plot_roc_curve(fpr_builtin, tpr_builtin, roc_auc_builtin, builtin_accuracy, "Built-in Logistic Regression", 'green')
plot_roc_curve(fpr_grid, tpr_grid, roc_auc_grid, grid_accuracy, "Grid Search Logistic Regression", 'red')
