import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from dgl.data.utils import load_graphs
import copy
import itertools
import json

train_idx = np.loadtxt('/home/hainguyen/Documents/GCN/data/amazon1/amazon_train_5.txt', dtype=int)
test_idx = np.loadtxt('/home/hainguyen/Documents/GCN/data/amazon1/amazon_test_20.txt', dtype=int)
val_idx = np.loadtxt('/home/hainguyen/Documents/GCN/data/amazon1/amazon_val_5.txt', dtype=int)
graph_list, label_dict = load_graphs('/home/hainguyen/Documents/GCN/data/amazon1/amazon')

device = 'cuda:0'
graph = graph_list[0].to(device)

features = graph.ndata['feature'].float().to(device)
labels = graph.ndata['label'].to(device)

train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)
val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")



class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, activation=F.relu):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, g, features):
        with g.local_scope():
            g.ndata['h'] = features
            g.ndata['ci'] = 1. / torch.sqrt(g.out_degrees().float().clamp(min=1))
            g.ndata['cj'] = 1. / torch.sqrt(g.in_degrees().float().clamp(min=1))
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
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_features, hidden_features, dropout, activation=F.relu)
        self.layer2 = GCNLayer(hidden_features, out_features, dropout, activation=None)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = self.layer2(g, x)
        return x

def train_and_eval(hidden_dim, dropout, lr, weight_decay, runs=3, epochs=200):
    test_f1s = []
    test_aucs = []
    for run in range(runs):
        print(f"Run {run + 1}/{runs}")
        model = GCN(in_features=features.shape[1],
                    hidden_features=hidden_dim,
                    out_features=2,
                    dropout=dropout).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        loss_fn = nn.CrossEntropyLoss()
        best_val = 0
        best_model = None

        for epoch in range(epochs):
            model.train()
            logits = model(graph, features)
            loss = loss_fn(logits[train_idx], labels[train_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(graph, features)
                pred = logits.argmax(1)
                val_f1 = f1_score(labels[val_idx].cpu().numpy(),
                                    pred[val_idx].cpu().numpy(), average='macro')
            
                val_probs = torch.softmax(logits[val_idx], dim=1).cpu().numpy()[:, 1]
                val_auc = roc_auc_score(labels[val_idx].cpu().numpy(), val_probs)
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
            if val_f1 > best_val:
                best_val = val_f1
                best_model = copy.deepcopy(model.state_dict())


        model.load_state_dict(best_model)
        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            pred = logits.argmax(1)
            test_f1 = f1_score(labels[test_idx].cpu().numpy(),
                                pred[test_idx].cpu().numpy(), average='macro')
            
            test_probs = torch.softmax(logits[test_idx], dim=1).cpu().numpy()[:, 1]
            test_auc = roc_auc_score(labels[test_idx].cpu().numpy(), test_probs)
        test_f1s.append(test_f1)
        test_aucs.append(test_auc)

    return np.mean(test_f1s), np.std(test_f1s), np.mean(test_aucs), np.std(test_aucs), best_model


avg_f1, std_f1, avg_auc, std_auc, best_model = train_and_eval(16, 0.5, 0.01, 5e-4, runs=3, epochs=200)
print(f"Test F1: {avg_f1:.4f} ± {std_f1:.4f}, Test AUC: {avg_auc:.4f} ± {std_auc:.4f}")