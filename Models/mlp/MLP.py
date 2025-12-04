import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from dgl.data.utils import load_graphs

GRAPH_PATH = "./dataset/weibo1/weibo"   
TRAIN_IDX_PATH = "./dataset/weibo1/train.txt"
VAL_IDX_PATH   = "./dataset/weibo1/val.txt"
TEST_IDX_PATH  = "./dataset/weibo1/test.txt"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LR = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 200
RUNS = 3
SEED = 42



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False

set_seed(SEED)



def load_index(path):
    arr = np.loadtxt(path, dtype=np.int64)
    return torch.tensor(arr, dtype=torch.long, device=DEVICE)
graph_list, _ = load_graphs(GRAPH_PATH)
graph = graph_list[0].to(DEVICE)

features = graph.ndata['feature'].float().to(DEVICE)
labels = graph.ndata['label'].long().to(DEVICE)

features = torch.tensor(features, dtype=torch.float32, device=DEVICE)
labels   = torch.tensor(labels, dtype=torch.long, device=DEVICE)

train_idx = load_index(TRAIN_IDX_PATH)
val_idx   = load_index(VAL_IDX_PATH)
test_idx  = load_index(TEST_IDX_PATH)

print(f"Device: {DEVICE}")
print(f"Train / Val / Test = {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
print(f"Features: {features.shape}")



class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x




@torch.no_grad()
def safe_auc(y_true, probs):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, probs)

@torch.no_grad()
def evaluate(model, idx):
    model.eval()
    logits = model(features[idx])

    preds = logits.argmax(dim=1).cpu().numpy()
    y_true = labels[idx].cpu().numpy()

    f1 = f1_score(y_true, preds, average="macro")
    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    auc = safe_auc(y_true, probs)

    return f1, auc



def train_and_eval(runs=RUNS):
    test_f1s, test_aucs = [], []

    for run in range(runs):
        print(f"\n===== Run {run+1} / {runs} =====")

        model = MLP(
            in_dim=features.shape[1],
            hidden_dim=64,
            out_dim=2,
            dropout=0.5
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.CrossEntropyLoss()

        best_f1 = -1
        best_state = None

        for epoch in range(EPOCHS):
            model.train()
            logits = model(features[train_idx])
            loss = loss_fn(logits, labels[train_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_f1, _ = evaluate(model, val_idx)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Loss {loss.item():.4f} | Val F1 {val_f1:.4f}")

        model.load_state_dict(best_state)
        test_f1, test_auc = evaluate(model, test_idx)

        print(f"[Run {run+1}] Test F1 = {test_f1:.4f}, AUC = {test_auc:.4f}")

        test_f1s.append(test_f1)
        test_aucs.append(test_auc if not np.isnan(test_auc) else 0)

    return test_f1s, test_aucs



f1_list, auc_list = train_and_eval()

print("\n===== Summary =====")
print(f"Test F1:  {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
print(f"Test AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
