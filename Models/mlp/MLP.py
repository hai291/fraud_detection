import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from dgl.data.utils import load_graphs
import copy
import numpy as np
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False

device = torch.device('cpu')
graph = graph.to(device)
features = graph.ndata['feature'].float().to(device)
labels = graph.ndata['label'].to(device)
train_idx = np.loadtxt('/home/hainguyen/Documents/mlp/data/yelp1/yelp_train.txt')
val_idx = np.loadtxt('/home/hainguyen/Documents/mlp/data/yelp1/yelp_val.txt')
test_idx = np.loadtxt('/home/hainguyen/Documents/mlp/data/yelp1/yelp_test.txt')


train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)
val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)



class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        super(SimpleMLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        
      
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),  
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
     
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=-1)



def evaluate_model(model, idx):
    model.eval()
    with torch.no_grad():
        logits = model(features[idx])
        pred = logits.argmax(1)
        f1 = f1_score(labels[idx].cpu().numpy(),
                      pred.cpu().numpy(),
                      average='macro')
        auc_score = roc_auc_score(labels[idx].cpu().numpy(),
                              model(features[idx]).exp()[:, 1].cpu().numpy())
    return f1, auc_score



test_f1_scores = []
test_auc_scores = []

for run in range(5):  
    
    set_seed(44 + run) 
    
    model = SimpleMLP(input_dim=features.shape[1],
                      hidden_dims=[8, 8],

                      output_dim=2,
                      dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.NLLLoss()

    best_val_f1 = 0


    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        logits = model(features[train_idx])
        loss = criterion(logits, labels[train_idx])
        
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Run {run + 1}, Epoch {epoch}, Loss: {loss.item():.4f}')

        val_f1 = evaluate_model(model, val_idx)

        if val_f1[0] > best_val_f1:
            best_val_f1 = val_f1[0]
            best_model = copy.deepcopy(model.state_dict())
            
     


  
   
    model.load_state_dict(best_model)
    test_f1, auc_score = evaluate_model(model, test_idx)
    print(f'Run {run + 1}, Best Val F1: {best_val_f1:.4f}, Test F1: {test_f1:.4f}, AUC: {auc_score:.4f}')
    test_f1_scores.append(test_f1)
    test_auc_scores.append(auc_score)

print(f'\nAverage Test F1 over 5 runs: {np.mean(test_f1_scores):.4f}')
print(f'Standard Deviation of Test F1 over 5 runs: {np.std(test_f1_scores):.4f}')
print(f'Average Test AUC over 5 runs: {np.mean(test_auc_scores):.4f}')
print(f'Standard Deviation of Test AUC over 5 runs: {np.std(test_auc_scores):.4f}')
