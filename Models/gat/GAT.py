import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
import time
from dgl.data.utils import load_graphs
import copy
import numpy as np

# Load data
train_idx = np.loadtxt('/home/hainguyen/Documents/mlp/data/weibo1/weibo_train_70.txt')
test_idx = np.loadtxt('/home/hainguyen/Documents/mlp/data/weibo1/weibo_test_20.txt')
val_idx = np.loadtxt('/home/hainguyen/Documents/mlp/data/weibo1/weibo_val_10.txt')
graph_list, label_dict = load_graphs('/home/hainguyen/Documents/mlp/data/weibo1/weibo')
graph = graph_list[0]

device = torch.device('cuda:0')
graph = graph.to(device)
features = graph.ndata['feature'].float().to(device)
labels = graph.ndata['label'].to(device)
train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)
val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)


class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0.6, alpha=0.2, residual=True):
        super(GATLayer, self).__init__()
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
        super(GATMultiHead, self).__init__()
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
        super(GAT, self).__init__()
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
        pred = logits[idx].argmax(1)
        f1 = f1_score(labels[idx].cpu().numpy(), pred.cpu().numpy(), average='macro')
        auc = roc_auc_score(labels[idx].cpu().numpy(),
                            F.softmax(logits[idx], dim=1)[:, 1].cpu().numpy())
    return f1, auc


start_time = time.ctime(time.time())
print(f'Start Time: {start_time}')

f1_scores = []
auc_scores = []

for run in range(5):
    print(f"\nRun {run+1}/5")
    model = GAT(in_feats=features.shape[1], hidden_feats=8, out_feats=2,
                num_heads=8, dropout=0.6, alpha=0.2, residual = False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    loss_fn = nn.CrossEntropyLoss()
    best_val_f1 = 0.0

    for epoch in range(200):
        model.train()
        logits = model(graph, features)
        loss = loss_fn(logits[train_idx], labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_f1, val_auc = evaluate(model, graph, features, labels, val_idx)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    f1_scor, auc = evaluate(model, graph, features, labels, test_idx)
    print(f"Test F1: {f1_scor:.4f}, Test AUC: {auc:.4f}")
    f1_scores.append(f1_scor)
    auc_scores.append(auc)

print("\n====================")
print(f"Average AUC over 5 runs: {np.mean(auc_scores) * 100:.2f} ± {np.std(auc_scores):.4f}")
print(f"F1 over 5 runs: {np.mean(f1_scores) * 100:.2f} ± {np.std(f1_scores):.4f}")


