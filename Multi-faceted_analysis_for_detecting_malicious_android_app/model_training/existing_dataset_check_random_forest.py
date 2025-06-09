import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt

# ---------- Step 1: Load and preprocess dataset ----------
dataset = pd.read_csv('../androZoo_dataset_analysis/TUANDROMD.csv').iloc[:, 1:]
#dataset = pd.read_csv('../androZoo_dataset_analysis/drebin.csv').iloc[:, 1:]
dataset.replace('?', np.nan, inplace=True)
dataset.dropna(inplace=True)

# Select only numeric columns
dataset = dataset.select_dtypes(include=[np.number])

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Step 2: Metric Helper ----------
results = []

def print_metrics(name, y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"True Positive Rate (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"AUC Score: {auc:.4f}")

    results.append({
        "Model": name,
        "Accuracy (%)": accuracy * 100,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC": auc
    })

# ---------- Step 3: Manual Random Forest ----------
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or len(y) < self.min_samples_split or depth == self.max_depth:
            return {'label': np.bincount(y.astype(int)).argmax()}

        best_split = self._find_best_split(X, y)
        if not best_split:
            return {'label': np.bincount(y.astype(int)).argmax()}

        left_tree = self._build_tree(X[best_split['left']], y[best_split['left']], depth + 1)
        right_tree = self._build_tree(X[best_split['right']], y[best_split['right']], depth + 1)
        return {
            'feature_index': best_split['feature_index'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def _find_best_split(self, X, y):
        best_gain = -1
        best_split = None
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left': left_mask,
                        'right': right_mask
                    }
        return best_split

    def _gini(self, y):
        classes = np.unique(y)
        return 1.0 - sum((np.sum(y == c) / len(y)) ** 2 for c in classes)

    def _information_gain(self, parent, left, right):
        p = float(len(left)) / len(parent)
        return self._gini(parent) - (p * self._gini(left) + (1 - p) * self._gini(right))

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def predict_proba(self, X):
        return np.array([
            1 if self._predict_sample(sample, self.tree) == 1 else 0 for sample in X
        ])

    def _predict_sample(self, sample, tree):
        if 'label' in tree:
            return tree['label']
        if sample[tree['feature_index']] <= tree['threshold']:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_estimators):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            print(f"Training tree {i + 1}/{self.n_estimators}...")
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=tree_preds)

    def predict_proba(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        probs = np.mean(tree_preds, axis=0)
        return probs

# ---------- Step 4: Train Manual RF ----------
print("\nTraining Manual Random Forest...")
manual_rf = RandomForest(n_estimators=10, max_depth=5)
manual_rf.fit(X_train, y_train)
manual_preds = manual_rf.predict(X_test)
manual_probs = manual_rf.predict_proba(X_test)
print_metrics("Manual Random Forest", y_test, manual_preds, manual_probs)

# ---------- Step 5: Built-in RF ----------
print("\nTraining Built-in Random Forest...")
builtin_rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=6, random_state=42)
builtin_rf.fit(X_train, y_train)
builtin_preds = builtin_rf.predict(X_test)
builtin_probs = builtin_rf.predict_proba(X_test)[:, 1]
print_metrics("Built-in Random Forest", y_test, builtin_preds, builtin_probs)

# ---------- Step 6: GridSearchCV RF ----------
print("\nTraining GridSearchCV Random Forest...")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 7, 15],
    'min_samples_split': [4, 6]
}
grid_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(grid_model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("\n===== Accuracy for Each Combination in GridSearchCV =====")
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df[['params', 'mean_test_score', 'rank_test_score']]
results_df = results_df.sort_values(by='rank_test_score')

for _, row in results_df.iterrows():
    print(f"Params: {row['params']}, Mean Accuracy: {row['mean_test_score']*100:.2f}%, Rank: {row['rank_test_score']}")

print("\n===== Grid Search Results =====")
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", f"{grid_search.best_score_ * 100:.2f}%")

best_model = grid_search.best_estimator_
grid_preds = best_model.predict(X_test)
grid_probs = best_model.predict_proba(X_test)[:, 1]
print_metrics("GridSearchCV Random Forest", y_test, grid_preds, grid_probs)

# ---------- Step 7: Print Summary Table ----------
summary_df = pd.DataFrame(results)
print("\n\n===== 📊 Final Model Comparison Table =====")
print(summary_df.to_string(index=False))

# ---------- Step 8: Plot ROC Curves ----------
fpr_manual, tpr_manual, _ = roc_curve(y_test, manual_probs)
fpr_builtin, tpr_builtin, _ = roc_curve(y_test, builtin_probs)
fpr_grid, tpr_grid, _ = roc_curve(y_test, grid_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_manual, tpr_manual, label=f'Manual RF AUC = {roc_auc_score(y_test, manual_probs):.4f}', color='blue')
plt.plot(fpr_builtin, tpr_builtin, label=f'Built-in RF AUC = {roc_auc_score(y_test, builtin_probs):.4f}', color='green')
plt.plot(fpr_grid, tpr_grid, label=f'GridSearch RF AUC = {roc_auc_score(y_test, grid_probs):.4f}', color='red')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves for Random Forest Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
