import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

# --- Load Datasets ---
real_data = pd.read_csv("androZoo_dataset_analysis/reduced_overall_static_dynamic_network_dataset.csv")
synthetic_data = pd.read_csv("androZoo_dataset_analysis/synthetic_static_dynamic_network_dataset.csv")

assert set(real_data.columns) == set(synthetic_data.columns), "Mismatch in columns"
features = [col for col in real_data.columns if col != 'label']

# --- Encode Categorical Columns ---
all_data = pd.concat([real_data, synthetic_data])
for col in features:
    if all_data[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(all_data[col].astype(str))
        real_data[col] = le.transform(real_data[col].astype(str))
        synthetic_data[col] = le.transform(synthetic_data[col].astype(str))

# --- Helper function to calculate TPR, FPR ---
def calculate_tpr_fpr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # Binary classification
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    else:
        # Multiclass: average TPR and FPR (macro)
        tpr_list, fpr_list = [], []
        for i in range(cm.shape[0]):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            tpr_list.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
            fpr_list.append(FP / (FP + TN) if (FP + TN) > 0 else 0)
        tpr = np.mean(tpr_list)
        fpr = np.mean(fpr_list)
    return tpr, fpr, cm

# --- Evaluation Containers ---
results = []

# --- Train on Synthetic, Test on Real (TSTR) ---
clf = RandomForestClassifier(random_state=42)
clf.fit(synthetic_data[features], synthetic_data['label'])
y_pred = clf.predict(real_data[features])

acc = accuracy_score(real_data['label'], y_pred)
prec = precision_score(real_data['label'], y_pred, average='macro', zero_division=0)
rec = recall_score(real_data['label'], y_pred, average='macro', zero_division=0)
f1 = f1_score(real_data['label'], y_pred, average='macro', zero_division=0)
tpr, fpr, cm_tstr = calculate_tpr_fpr(real_data['label'], y_pred)
results.append(['TSTR', acc, prec, rec, f1, tpr, fpr])

# --- Train on Real, Test on Synthetic (TRTS) ---
clf = RandomForestClassifier(random_state=42)
clf.fit(real_data[features], real_data['label'])
y_pred = clf.predict(synthetic_data[features])

acc = accuracy_score(synthetic_data['label'], y_pred)
prec = precision_score(synthetic_data['label'], y_pred, average='macro', zero_division=0)
rec = recall_score(synthetic_data['label'], y_pred, average='macro', zero_division=0)
f1 = f1_score(synthetic_data['label'], y_pred, average='macro', zero_division=0)
tpr, fpr, cm_trts = calculate_tpr_fpr(synthetic_data['label'], y_pred)
results.append(['TRTS', acc, prec, rec, f1, tpr, fpr])

# --- Cross-validation (Synthetic + Real Mix) with detailed metrics ---
mixed_data = pd.concat([real_data, synthetic_data])
X = mixed_data[features]
y = mixed_data['label']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores, prec_scores, rec_scores, f1_scores, tpr_scores, fpr_scores = [], [], [], [], [], []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc_scores.append(accuracy_score(y_test, y_pred))
    prec_scores.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
    rec_scores.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

    tpr, fpr, _ = calculate_tpr_fpr(y_test, y_pred)
    tpr_scores.append(tpr)
    fpr_scores.append(fpr)

results.append([
    'Cross-Validation',
    np.mean(acc_scores),
    np.mean(prec_scores),
    np.mean(rec_scores),
    np.mean(f1_scores),
    np.mean(tpr_scores),
    np.mean(fpr_scores)
])

# --- Classifier Two-Sample Test (CTST) ---
real_sample = real_data.copy().assign(source=0)
synthetic_sample = synthetic_data.copy().assign(source=1)

combined = pd.concat([real_sample, synthetic_sample])
X = combined[features]
y = combined['source']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
clf = GradientBoostingClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
tpr, fpr, cm_ctst = calculate_tpr_fpr(y_test, y_pred)
results.append(['CTST', acc, prec, rec, f1, tpr, fpr])

# --- Results DataFrame ---
results_df = pd.DataFrame(results, columns=['Test', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'TPR', 'FPR'])
print(results_df)

# --- Plot Evaluation Metrics ---
plt.figure(figsize=(14, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'TPR', 'FPR']

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    sns.barplot(data=results_df, x='Test', y=metric)
    plt.title(metric)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.suptitle("Evaluation Metrics for Real vs Synthetic Dataset Analysis", fontsize=16, y=1.02)
plt.show()
