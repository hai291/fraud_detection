# Base Research Repository

## Cấu trúc thư mục

```
base-research-repo/           
├── config          # Các file cấu hình cho training
│   ├── model_config.yaml
│   └── training_config.yaml
├── dataset
│   ├── build_graph
│   │   └── process_node_address_all.py     # xử lý transaction records thành đồ thị
│   ├── name.py
│   ├── raw         # Thư mục chứa dữ liệu thô
│   ├── README.md           # Mô tả các dataset    
│   └── train_test_split.py         # Chia dữ liệu train_test
├── Models
│   ├── gat         # Mô hình GAT
│   │   ├── GAT.py
│   │   └── pyproject.toml
│   ├── gcn         # Mô hình GCN
│   │   ├── GCN.py
│   │   └── pyproject.toml
│   └── mlp         # Mô hình MLP
│       ├── MLP.py
│       └── pyproject.toml
```
``` bash
# Bulding graph
cd dataset/build_graph
python process_node_address_all.py

# Split train_test_split 
python main.py

# Training
cd ..
python GCN.py
python GAT.py
python MLP.py