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

def process_trans_id(type_, class_name, address_to_id, next_addr_id, unique_hash):

    if class_name == 'phisher':
        path = f'/home/hains/Data_B4E/phish_trans/{class_name}_transaction_{type_}.csv'
    else:
        path = f'/home/hains/Data_B4E/normal_trans/{class_name}_eoa_transaction_{type_}_slice_1000K.csv'

    df = pd.read_csv(path, names=HEADER)
    df = df.sort_values(by='block_timestamp')
    print(f"Loaded {len(df)} transactions from {path}")

    window_size = 14 * 24 * 3600  
    active_period = 3 * 3600    

    edges = []
    label_dict = {}     
    feature_dict = {}    
    node_not_edges = []  


    from_map = defaultdict(list)   
    to_map = defaultdict(list)    
    window_tx = []             


    if df.shape[0] == 0:
        return edges, label_dict, feature_dict, address_to_id, next_addr_id, unique_hash, node_not_edges

    start_time = df['block_timestamp'].iloc[0]
    window_end = start_time + window_size

    for row in df.itertuples(index=False):

        if row.hash in unique_hash:
            continue
        unique_hash.add(row.hash)


        from_id, next_addr_id = get_address_id(row.from_address, address_to_id, next_addr_id)
        to_id, next_addr_id = get_address_id(row.to_address, address_to_id, next_addr_id)

        label_value = 1 if class_name == 'phisher' else 0
        feature_value = [from_id, to_id, float(row.value) / 1e12, row.block_timestamp]
        label_dict[row.hash] = label_value
        feature_dict[row.hash] = feature_value


        window_tx.append(row.hash)

        t = row.block_timestamp


        if t > window_end:

            for addr in to_map:
                if addr in from_map:
                    for tx_in, t_in_ts in to_map[addr]:
                        for tx_out, t_out_ts in from_map[addr]:
  
                            if t_in_ts <= t_out_ts:
                                edges.append((tx_in, tx_out))



            node_not_edges.append(list(window_tx))

            from_map.clear()
            to_map.clear()
            window_tx.clear()
            start_time = t
            window_end = start_time + window_size




        if t - start_time <= active_period:
            from_map[from_id].append((row.hash, t))
            to_map[to_id].append((row.hash, t))



    if to_map:
        for addr in to_map:
            if addr in from_map:
                for tx_in, t_in_ts in to_map[addr]:
                    for tx_out, t_out_ts in from_map[addr]:
                        if t_in_ts <= t_out_ts:
                            edges.append((tx_in, tx_out))


    if window_tx:
        node_not_edges.append(list(window_tx))

    print(f"→ Done {class_name} {type_}: {len(edges)} edges, {len(label_dict)} transactions (nodes)")
    return edges, label_dict, feature_dict, address_to_id, next_addr_id, unique_hash, node_not_edges


def build_graph(label_dict, feature_dict, address_to_id, edge_list, node_not_edges, save_path):

    tx_list = list(label_dict.keys())
    tx_to_id = {tx: idx for idx, tx in enumerate(tx_list)}
    num_nodes = len(tx_list)


    src = []
    dst = []
    for u_hash, v_hash in edge_list:
        if u_hash in tx_to_id and v_hash in tx_to_id:
            src.append(tx_to_id[u_hash])
            dst.append(tx_to_id[v_hash])


    graph = dgl.graph((src, dst), num_nodes=num_nodes)


    label_tensor = torch.tensor([label_dict[tx] for tx in tx_list], dtype=torch.long)
    feature_tensor = torch.tensor([feature_dict[tx] for tx in tx_list], dtype=torch.float32)


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
    print(f"\nGraph built with {num_nodes} nodes, {len(src)} edges.")
    print(f"Saved to: {save_path}")


    indeg = graph.in_degrees().numpy()
    outdeg = graph.out_degrees().numpy()
    isolated = int(((indeg + outdeg) == 0).sum())
    print(f"Isolated nodes (degree 0): {isolated} / {num_nodes}")

    return graph


if __name__ == '__main__':


    address_to_id = {}
    next_addr_id = 0
    unique_hash = set()
    all_edges, all_labels, all_features, node_not_edges = [], {}, {}, []

    for type_, cls in [('in', 'phisher'), ('out', 'phisher'), ('in', 'normal'), ('out', 'normal')]:
        edges, lbl, feat, address_to_id, next_addr_id, unique_hash, not_edges = process_trans_id(
            type_, cls, address_to_id, next_addr_id, unique_hash
        )
        all_edges.extend(edges)
        all_labels.update(lbl)
        all_features.update(feat)
        node_not_edges.extend(not_edges)

    with open('/home/hains/Data_B4E/edge_list_total.pkl', 'wb') as f:
        pkl.dump(all_edges, f)

    build_graph(all_labels, all_features, address_to_id, all_edges, node_not_edges, '/home/hains/Data_B4E/b4e_trans_3h')
