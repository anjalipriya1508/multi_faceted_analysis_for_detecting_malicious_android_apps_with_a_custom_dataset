import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

# === Load and preprocess ===
df = pd.read_csv("../androZoo_dataset_analysis/reduced_overall_static_dynamic_network_dataset.csv").iloc[:, 1:]

label_col = 'label'
feature_cols = [col for col in df.columns if col != label_col]

X = df[feature_cols].values.astype(np.float32)
y = df[label_col].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === Few-shot config ===
N_WAY = 2
K_SHOT = 5
Q_QUERY = 15
EMBEDDING_DIM = 64

# === Embedding network ===
class MLPEmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)

model = MLPEmbeddingNet(X.shape[1], embedding_dim=EMBEDDING_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === Few-shot episode sampler ===
def sample_episode(X, y, n_way, k_shot, q_query):
    class_counts = Counter(y)
    eligible_classes = [cls for cls, count in class_counts.items() if count >= (k_shot + q_query)]
    if len(eligible_classes) < n_way:
        raise ValueError("Not enough eligible classes")

    selected_classes = random.sample(eligible_classes, n_way)
    support_x, support_y, query_x, query_y = [], [], [], []

    for i, cls in enumerate(selected_classes):
        indices = np.where(y == cls)[0]
        chosen = np.random.choice(indices, size=(k_shot + q_query), replace=False)
        support_x.append(X[chosen[:k_shot]])
        support_y += [i] * k_shot
        query_x.append(X[chosen[k_shot:]])
        query_y += [i] * q_query

    return (
        torch.tensor(np.vstack(support_x), dtype=torch.float32),
        torch.tensor(support_y, dtype=torch.long),
        torch.tensor(np.vstack(query_x), dtype=torch.float32),
        torch.tensor(query_y, dtype=torch.long)
    )

# === Prototypical Network ===
def compute_prototypes(embeddings, labels, n_way):
    return torch.stack([embeddings[labels == i].mean(0) for i in range(n_way)])

def prototypical_loss(prototypes, queries, query_labels):
    dists = torch.cdist(queries, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p_y, query_labels)
    acc = (log_p_y.argmax(1) == query_labels).float().mean()
    return loss, acc

# === Training ===
EPOCHS = 1000
print(f"\n[INFO] Starting training for {EPOCHS} episodes...")
for epoch in range(1, EPOCHS + 1):
    try:
        sx, sy, qx, qy = sample_episode(X_train, y_train, N_WAY, K_SHOT, Q_QUERY)
        model.train()
        support_emb = model(sx)
        query_emb = model(qx)
        prototypes = compute_prototypes(support_emb, sy, N_WAY)
        loss, acc = prototypical_loss(prototypes, query_emb, qy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f} | Accuracy: {acc.item() * 100:.2f}%")
    except ValueError as e:
        print(f"[Epoch {epoch}] Skipped: {e}")

# === Evaluation ===
accuracies = []
EPISODES = 50
K_EVAL = 10
Q_EVAL = 15

print(f"\n[INFO] Starting evaluation for {EPISODES} episodes...")
for _ in range(EPISODES):
    try:
        sx, sy, qx, qy = sample_episode(X_test, y_test, N_WAY, K_EVAL, Q_EVAL)
        with torch.no_grad():
            support_emb = model(sx)
            query_emb = model(qx)
            prototypes = compute_prototypes(support_emb, sy, N_WAY)
            dists = torch.cdist(query_emb, prototypes)
            preds = torch.argmin(dists, dim=1)
            acc = (preds == qy).float().mean().item()
            accuracies.append(acc)
    except ValueError:
        continue

# === Summary Report ===
if accuracies:
    avg_acc = np.mean(accuracies) * 100
    print("\n=== Few-Shot Evaluation Summary ===")
    print(f"Number of Episodes      : {len(accuracies)}")
    print(f"N-WAY                   : {N_WAY}")
    print(f"K-SHOT (train)          : {K_SHOT}")
    print(f"K-EVAL (test support)   : {K_EVAL}")
    print(f"Q-QUERY (train)         : {Q_QUERY}")
    print(f"Q-EVAL (test query)     : {Q_EVAL}")
    print(f"Embedding Dimension     : {EMBEDDING_DIM}")
    print(f"Average Test Accuracy   : {avg_acc:.2f}%")
else:
    print("Not enough data to evaluate.")
