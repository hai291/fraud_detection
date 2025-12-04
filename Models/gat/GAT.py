# weibo_gat_train.py
import os
import time
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from dgl.data.utils import load_graphs

GRAPH_PATH = './dataset/b4eaddress1/b4eaddress'    
TRAIN_IDX_PATH = './dataset/b4eaddress1/b4eaddress_train.txt'
VAL_IDX_PATH = './dataset/b4eaddress1/b4eaddress_val.txt'
TEST_IDX_PATH = './dataset/b4eaddress1/b4eaddress_test.txt'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED = 42
NUM_RUNS = 5
EPOCHS = 200
LR = 0.005
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


train_idx_np = load_idx(TRAIN_IDX_PATH)
val_idx_np = load_idx(VAL_IDX_PATH)
test_idx_np = load_idx(TEST_IDX_PATH)

graph_list, label_dict = load_graphs(GRAPH_PATH)
graph = graph_list[0]

graph = graph.to(DEVICE)                 
features = graph.ndata['feature'].float().to(DEVICE)
labels = graph.ndata['label'].long().to(DEVICE)

train_idx = torch.from_numpy(train_idx_np).long().to(DEVICE)
val_idx = torch.from_numpy(val_idx_np).long().to(DEVICE)
test_idx = torch.from_numpy(test_idx_np).long().to(DEVICE)


class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0.6, alpha=0.2, residual=True):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.dropout = dropout
        self.residual = residual

        self.W = nn.Parameter(torch.empty(in_feats, out_feats))
        self.a = nn.Parameter(torch.empty(2 * out_feats, 1))
        self.leakyrelu = nn.LeakyReLU(alpha)

        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.W, gain=gain)
        nn.init.xavier_normal_(self.a, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['Wh'], edges.dst['Wh']], dim=1)
        e = self.leakyrelu(torch.matmul(z2, self.a))
        return {'e': e}

    def message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['Wh'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        h_in = h
        h = F.dropout(h, p=self.dropout, training=self.training)
        Wh = torch.matmul(h, self.W)
        g.ndata['Wh'] = Wh
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h_out = g.ndata.pop('h')

        if self.residual:
            h_res = self.res_fc(h_in) if self.res_fc is not None else 0
            h_out = h_out + h_res

        return h_out

class GATMultiHead(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, merge='cat',
                 dropout=0.6, alpha=0.2, residual=False):
        super().__init__()
        self.heads = nn.ModuleList([
            GATLayer(in_feats, out_feats, dropout, alpha, residual=residual)
            for _ in range(num_heads)
        ])
        self.merge = merge

    def forward(self, g, h):
        out = [head(g, h) for head in self.heads]
        if self.merge == 'cat':
            return torch.cat(out, dim=1)
        elif self.merge == 'mean':
            return torch.mean(torch.stack(out), dim=0)
        else:
            return out[0]

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads,
                 dropout=0.6, alpha=0.2, residual=True):
        super().__init__()
        self.gat1 = GATMultiHead(in_feats, hidden_feats, num_heads,
                                 merge='cat', dropout=dropout, alpha=alpha,
                                 residual=residual)
        self.gat2 = GATMultiHead(hidden_feats * num_heads, out_feats, 1,
                                 merge='mean', dropout=dropout, alpha=alpha,
                                 residual=residual)

    def forward(self, g, h):
        h = self.gat1(g, h)
        h = F.elu(h)
        h = self.gat2(g, h)
        return h


def evaluate(model, g, features, labels, idx):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        probs = F.softmax(logits[idx], dim=1)
        preds = logits[idx].argmax(1)
        y_true = labels[idx].cpu().numpy()
        y_pred = preds.cpu().numpy()
        f1 = f1_score(y_true, y_pred, average='macro')
      
        try:
            auc = roc_auc_score(y_true, probs[:, 1].cpu().numpy())
        except Exception:
            auc = float('nan')
    return f1, auc


print("Start training on device:", DEVICE)
start_time = time.ctime(time.time())
print("Start Time:", start_time)

f1_scores = []
auc_scores = []

for run in range(NUM_RUNS):
    print(f"\nRun {run+1}/{NUM_RUNS}")
    model = GAT(in_feats=features.shape[1], hidden_feats=8, out_feats=2,
                num_heads=8, dropout=0.6, alpha=0.2, residual=False).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        logits = model(graph, features)
        loss = loss_fn(logits[train_idx], labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_f1, val_auc = evaluate(model, graph, features, labels, val_idx)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc}")

        val_f1, _ = evaluate(model, graph, features, labels, val_idx)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())


    if best_state is not None:
        model.load_state_dict(best_state)
    test_f1, test_auc = evaluate(model, graph, features, labels, test_idx)
    print(f"Test F1: {test_f1:.4f}, Test AUC: {test_auc}")
    f1_scores.append(test_f1)
    auc_scores.append(test_auc if not np.isnan(test_auc) else 0.0)


print("\n====================")
print(f"Average AUC over {NUM_RUNS} runs: {np.mean(auc_scores) * 100:.2f} ± {np.std(auc_scores):.4f}")
print(f"Average F1 over {NUM_RUNS} runs: {np.mean(f1_scores) * 100:.2f} ± {np.std(f1_scores):.4f}")
