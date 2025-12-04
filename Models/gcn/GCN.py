# amazon_gcn_train.py
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from dgl.data.utils import load_graphs


GRAPH_PATH = './dataset/weibo1/weibo'    
TRAIN_IDX_PATH = './dataset/weibo1/weibo_train.txt'
VAL_IDX_PATH = './dataset/weibo1/weibo_val.txt'
TEST_IDX_PATH = './dataset/weibo1/weibo_test.txt'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 42

RUNS = 3
EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_idx(path):
    arr = np.loadtxt(path, dtype=np.int64)
    if np.isscalar(arr):
        arr = np.array([int(arr)], dtype=np.int64)
    return arr

def safe_auc(y_true, probs):

    labels = np.unique(y_true)
    if len(labels) < 2:
        return float('nan')
    try:
        return roc_auc_score(y_true, probs)
    except Exception:
        return float('nan')


train_idx_np = load_idx(TRAIN_IDX_PATH)
val_idx_np = load_idx(VAL_IDX_PATH)
test_idx_np = load_idx(TEST_IDX_PATH)

graph_list, _ = load_graphs(GRAPH_PATH)
graph = graph_list[0].to(DEVICE)

features = graph.ndata['feature'].float().to(DEVICE)
labels = graph.ndata['label'].long().to(DEVICE)

train_idx = torch.from_numpy(train_idx_np).long().to(DEVICE)
val_idx = torch.from_numpy(val_idx_np).long().to(DEVICE)
test_idx = torch.from_numpy(test_idx_np).long().to(DEVICE)

print(f"Device: {DEVICE}")
print(f"Train/Val/Test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
print(f"Num nodes: {graph.num_nodes()}, Num edges: {graph.num_edges()}")


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, activation=F.relu):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, g, features):
        with g.local_scope():
            h = self.dropout(features)
            g.ndata['h'] = h

            ci = 1. / torch.sqrt(g.out_degrees().float().clamp(min=1)).to(h.device)
            cj = 1. / torch.sqrt(g.in_degrees().float().clamp(min=1)).to(h.device)
            g.ndata['ci'] = ci
            g.ndata['cj'] = cj
            g.update_all(self.mfunc, self.rfunc)
            x = g.ndata.pop('h')
            x = self.linear(x)
            if self.activation is not None:
                x = self.activation(x)
            return x

    def mfunc(self, edges):
        return {'m': edges.src['h'], 'ci': edges.src['ci']}

    def rfunc(self, nodes):
        m = nodes.mailbox['m']                     
        ci = nodes.mailbox['ci'].unsqueeze(-1)    
        h = torch.sum(m * ci, dim=1) * nodes.data['cj'].unsqueeze(-1)
        return {'h': h}

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.5):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden_features, dropout, activation=F.relu)
        self.layer2 = GCNLayer(hidden_features, out_features, dropout, activation=None)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = self.layer2(g, x)
        return x


def evaluate(model, g, features, labels, idx):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        probs = F.softmax(logits[idx], dim=1)[:, 1].cpu().numpy() if logits.size(1) > 1 else np.zeros(len(idx))
        preds = logits[idx].argmax(1).cpu().numpy()
        y_true = labels[idx].cpu().numpy()
        f1 = f1_score(y_true, preds, average='macro')
        auc = safe_auc(y_true, probs)
    return f1, auc

def train_and_eval(hidden_dim=16, dropout=0.5, lr=LR, weight_decay=WEIGHT_DECAY, runs=RUNS, epochs=EPOCHS):
    test_f1s, test_aucs = [], []
    best_states = []
    for run in range(runs):
        print(f"\n=== Run {run+1}/{runs} ===")
        model = GCN(in_features=features.shape[1],
                    hidden_features=hidden_dim,
                    out_features=2,
                    dropout=dropout).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        best_val_f1 = -1.0
        best_state = None

        for epoch in range(epochs):
            model.train()
            logits = model(graph, features)
            loss = loss_fn(logits[train_idx], labels[train_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_f1, val_auc = evaluate(model, graph, features, labels, val_idx)
                print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}  Val F1: {val_f1:.4f}  Val AUC: {val_auc}")

            val_f1, _ = evaluate(model, graph, features, labels, val_idx)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())

        # load best and evaluate on test
        if best_state is not None:
            model.load_state_dict(best_state)
        test_f1, test_auc = evaluate(model, graph, features, labels, test_idx)
        print(f"Run {run+1} Test F1: {test_f1:.4f}  Test AUC: {test_auc}")
        test_f1s.append(test_f1)
        test_aucs.append(test_auc if not np.isnan(test_auc) else 0.0)
        best_states.append(best_state)

    return np.mean(test_f1s), np.std(test_f1s), np.mean(test_aucs), np.std(test_aucs), best_states


avg_f1, std_f1, avg_auc, std_auc, best_states = train_and_eval(hidden_dim=16, dropout=0.5, lr=LR, weight_decay=WEIGHT_DECAY, runs=RUNS, epochs=EPOCHS)
print("\n===== Summary =====")
print(f"Test F1: {avg_f1:.4f} ± {std_f1:.4f}")
print(f"Test AUC: {avg_auc:.4f} ± {std_auc:.4f}")


if best_states and best_states[0] is not None:
    torch.save(best_states[0], "best_model_run1.pt")
    print("Saved best_model_run1.pt")
