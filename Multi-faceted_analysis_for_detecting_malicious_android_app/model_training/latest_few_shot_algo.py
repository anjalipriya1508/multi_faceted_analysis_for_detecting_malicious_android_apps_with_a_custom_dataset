import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False*")

# === Load and preprocess ===
df = pd.read_csv("../androZoo_dataset_analysis/overall_static_dynamic_network_dataset.csv").iloc[:, 1:]
label_col = 'label'
cat_cols = [col for col in df.columns if col != label_col]

encoders = {col: LabelEncoder().fit(df[col]) for col in cat_cols}
for col in cat_cols:
    df[col] = encoders[col].transform(df[col])

X = df[cat_cols].values
y = df[label_col].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Few-shot params ===
N_WAY, K_SHOT, Q_QUERY, EPOCHS = 2, 20, 20, 5000
num_classes_per_col = [len(encoders[col].classes_) for col in cat_cols]

# === Transformer-based Few-shot Embedding Network ===
class TransformerEmbeddingNet(nn.Module):
    def __init__(self, num_classes_per_col, max_embedding_dim=32, final_embedding_dim=64):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(n_classes, min(max_embedding_dim, (n_classes + 1) // 2))
            for n_classes in num_classes_per_col
        ])
        self.embed_dims = [emb.embedding_dim for emb in self.emb_layers]
        self.total_dim = sum(self.embed_dims)
        self.pad_dim = max(self.embed_dims)

        self.projectors = nn.ModuleList([
            nn.Linear(dim, self.pad_dim) if dim != self.pad_dim else nn.Identity()
            for dim in self.embed_dims
        ])
        self.n_features = len(num_classes_per_col)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.pad_dim, nhead=4, batch_first=True,
            dim_feedforward=128, activation="gelu", dropout=0.1, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_features * self.pad_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, final_embedding_dim),
            nn.LayerNorm(final_embedding_dim)
        )

    def forward(self, x):
        embs = [proj(emb(x[:, i])) for i, (emb, proj) in enumerate(zip(self.emb_layers, self.projectors))]
        x_emb = torch.stack(embs, dim=1)  # [B, F, D]
        x_trans = self.encoder(x_emb)     # [B, F, D]
        return self.head(x_trans)

# === Few-shot utilities ===
def sample_episode(X, y, n_way, k_shot, q_query):
    class_counts = Counter(y)
    eligible = [cls for cls, count in class_counts.items() if count >= (k_shot + q_query)]
    if len(eligible) < n_way:
        raise ValueError("Not enough eligible classes.")
    chosen_classes = random.sample(eligible, n_way)
    sx, sy, qx, qy = [], [], [], []
    for i, cls in enumerate(chosen_classes):
        cls_idx = np.where(y == cls)[0]
        chosen = np.random.choice(cls_idx, k_shot + q_query, replace=False)
        sx.append(X[chosen[:k_shot]])
        qx.append(X[chosen[k_shot:]])
        sy += [i] * k_shot
        qy += [i] * q_query
    return (
        torch.tensor(np.vstack(sx), dtype=torch.long),
        torch.tensor(sy, dtype=torch.long),
        torch.tensor(np.vstack(qx), dtype=torch.long),
        torch.tensor(qy, dtype=torch.long),
    )

def compute_prototypes(embeddings, labels, n_way):
    return torch.stack([embeddings[labels == i].mean(0) for i in range(n_way)])

def prototypical_loss(prototypes, queries, query_labels):
    dists = torch.cdist(queries, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p_y, query_labels)
    acc = (log_p_y.argmax(1) == query_labels).float().mean()
    return loss, acc

# === Training loop ===
def train_model(model, X_train, y_train):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(1, EPOCHS + 1):
        try:
            sx, sy, qx, qy = sample_episode(X_train, y_train, N_WAY, K_SHOT, Q_QUERY)
            model.train()
            support = model(sx)
            query = model(qx)
            prototypes = compute_prototypes(support, sy, N_WAY)
            loss, acc = prototypical_loss(prototypes, query, qy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 200 == 0:
                print(f"[Epoch {epoch}] Loss: {loss.item():.4f} | Acc: {acc.item()*100:.2f}%")
        except ValueError:
            continue

# === Evaluation with FNR included ===
def evaluate_metrics(model, X_test, y_test):
    model.eval()
    sx, sy, qx, qy = sample_episode(X_test, y_test, N_WAY, K_SHOT, Q_QUERY)
    with torch.no_grad():
        support = model(sx)
        query = model(qx)
        prototypes = compute_prototypes(support, sy, N_WAY)
        dists = torch.cdist(query, prototypes)
        preds = torch.argmin(dists, dim=1).numpy()
        truth = qy.numpy()
        
        f1 = f1_score(truth, preds)
        acc = accuracy_score(truth, preds)
        recall = recall_score(truth, preds)  # TPR
        cm = confusion_matrix(truth, preds)
        
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            fpr = fnr = 0.0
        
        return f1, acc, recall, fpr, fnr

# === t-SNE Visualization ===
def tsne_plot(model, X, y, title, save_path):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.long)
        embeddings = model(X_tensor).cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 7))
    for label in np.unique(y):
        idx = y == label
        plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], 
                    label='Benign' if label == 0 else 'Malicious',
                    alpha=0.6, s=50)
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.show()

# === Instantiate and Train ===
transformer_model = TransformerEmbeddingNet(num_classes_per_col)
print("Training Transformer-Based Model...")
train_model(transformer_model, X_train, y_train)

# === Evaluate and Visualize ===
tsne_plot(transformer_model, X_test, y_test, "Transformer Embeddings t-SNE", "transformer_tsne.png")
f1, acc, recall, fpr, fnr = evaluate_metrics(transformer_model, X_test, y_test)

# === Final Output ===
print("\n--- Transformer-Based Model Performance ---")
print(f"F1 Score     : {f1:.4f}")
print(f"Accuracy     : {acc:.4f}")
print(f"Recall (TPR) : {recall:.4f}")
print(f"FPR          : {fpr:.4f}")
print(f"FNR          : {fnr:.4f}")
