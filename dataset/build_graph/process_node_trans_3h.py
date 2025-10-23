import pandas as pd
import pickle as pkl
import dgl
import torch
import random
from collections import defaultdict

HEADER = [
    'hash','nonce','block_hash','block_number','transaction_index',
    'from_address','to_address','value','gas','gas_price',
    'input','block_timestamp','max_fee_per_gas',
    'max_priority_fee_per_gas','transaction_type'
]

def load_phish_set(path):
    with open(path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

def get_address_id(address, address_to_id, next_addr_id):
    if address not in address_to_id:
        address_to_id[address] = next_addr_id
        next_addr_id += 1
    return address_to_id[address], next_addr_id

def process_trans_id(type_, class_name, phish_set, address_to_id, next_addr_id, unique_hash):

    if class_name == 'phisher':
        path = f'/home/hainguyen/fraud-detection-credit/dataset/raw/phish_trans/{class_name}_transaction_{type_}.csv'
    else:
        path = f'/home/hainguyen/fraud-detection-credit/dataset/raw/normal_trans/{class_name}_eoa_transaction_{type_}_slice_1000K.csv'

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} transactions from {path}")

    start_time = df['block_timestamp'].iloc[0]
    first_index = 0
    edges = []
    label_dict = {}
    feature_dict = {}

    while True:

        found_3h = False
        for row in df.iloc[first_index:].itertuples(index=True):
            if row.block_timestamp - start_time >= 10800: 
                last_index = row.Index
                last_timestamp = row.block_timestamp
                three_hour_slice = df.iloc[first_index:last_index]

                from_map = defaultdict(list) 
                to_map = defaultdict(list)   

                for tx in three_hour_slice.itertuples(index=True):
                    if tx.hash in unique_hash:
                        continue
                    unique_hash.add(tx.hash)

                    from_id, next_addr_id = get_address_id(tx.from_address, address_to_id, next_addr_id)
                    to_id, next_addr_id = get_address_id(tx.to_address, address_to_id, next_addr_id)

                    label_value = 1 if tx.from_address in phish_set or tx.to_address in phish_set else 0
                    feature_value = [from_id, to_id, float(tx.value) / 1e12, tx.block_timestamp]

                    label_dict[tx.hash] = label_value
                    feature_dict[tx.hash] = feature_value

                    from_map[from_id].append(tx.hash)
                    to_map[to_id].append(tx.hash)

    
                for _, tx_hashes in from_map.items():
                    for tx_hash in tx_hashes:
  
                        to_id = feature_dict[tx_hash][1]
                        if to_id in from_map:
                            for next_tx in from_map[to_id]:
                                edges.append((tx_hash, next_tx))

                first_index = last_index
                start_time = last_timestamp
                found_3h = True
                break

        if not found_3h:
            break


        found_2w = False
        for row in df.iloc[first_index:].itertuples(index=True):
            if row.block_timestamp - start_time >= 1209600:
                first_index = row.Index
                start_time = row.block_timestamp
                found_2w = True
                break
        if not found_2w:
            break

    print(f"→ Done {class_name} {type_}: {len(edges)} edges, {len(label_dict)} transactions")
    return edges, label_dict, feature_dict, address_to_id, next_addr_id, unique_hash

def build_graph(label_dict, feature_dict, address_to_id, edge_list, save_path):
    tx_to_id = {tx: idx for idx, tx in enumerate(label_dict.keys())}
    num_nodes = len(tx_to_id)

    source_nodes = [tx_to_id[u] for u, v in edge_list if u in tx_to_id and v in tx_to_id]
    destination_nodes = [tx_to_id[v] for u, v in edge_list if u in tx_to_id and v in tx_to_id]

    graph = dgl.graph((source_nodes, destination_nodes), num_nodes=num_nodes)

    label_tensor = torch.tensor([label_dict[tx] for tx in label_dict], dtype=torch.long)
    feature_tensor = torch.tensor([feature_dict[tx] for tx in feature_dict], dtype=torch.float32)
    indices = list(range(num_nodes))
    random.shuffle(indices)
    train_split = int(0.7 * num_nodes)
    val_split = int(0.8 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_split]] = True
    val_mask[indices[train_split:val_split]] = True
    test_mask[indices[val_split:]] = True

    graph.ndata['label'] = label_tensor
    graph.ndata['feature'] = feature_tensor
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    dgl.save_graphs(save_path, [graph])
    print(f"\nGraph built with {num_nodes} nodes, {len(edge_list)} edges.")
    print(f"Saved to: {save_path}")
    return graph

if __name__ == '__main__':
    phish_set = load_phish_set('/home/hainguyen/fraud-detection-credit/dataset/raw/phisher_account.txt')

    address_to_id = {}
    next_addr_id = 0
    unique_hash = set()

    all_edges, all_labels, all_features = [], {}, {}

    for type_, cls in [('in', 'phisher'), ('out', 'phisher'), ('in', 'normal'), ('out', 'normal')]:
        edges, lbl, feat, address_to_id, next_addr_id, unique_hash = process_trans_id(
            type_, cls, phish_set, address_to_id, next_addr_id, unique_hash
        )
        all_edges.extend(edges)
        all_labels.update(lbl)
        all_features.update(feat)


    build_graph(all_labels, all_features, address_to_id, all_edges, '/home/hainguyen/fraud-detection-credit/dataset/processed/b4e_trans_3h')
