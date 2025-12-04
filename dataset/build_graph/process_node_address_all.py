import pandas as pd
import dgl
import torch
import pickle as pkl
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


def process_address_graph(type_, class_name, phish_set, address_to_id, next_addr_id):

    if class_name == 'phisher':
        path = f'./raw/phish_trans/{class_name}_transaction_{type_}.csv'
    else:
        path = f'./normal_trans/{class_name}_eoa_transaction_{type_}_slice_1000K.csv'

    df = pd.read_csv(path)
    df = df.sort_values('block_timestamp').reset_index(drop=True)
    print(f'Loaded {len(df)} transactions from {path}')

    edges = []
    out_count = defaultdict(int)
    in_count = defaultdict(int)
    out_value = defaultdict(float)
    in_value = defaultdict(float)
    last_ts = defaultdict(int)

    for tx in df.itertuples(index=False):
        from_addr = str(tx.from_address)
        to_addr = str(tx.to_address)
        val = float(tx.value) if tx.value not in (None, '', 'NaN') else 0.0
        ts = int(tx.block_timestamp)

        from_id, next_addr_id = get_address_id(from_addr, address_to_id, next_addr_id)
        to_id, next_addr_id = get_address_id(to_addr, address_to_id, next_addr_id)

        edges.append((from_id, to_id))

        out_count[from_id] += 1
        in_count[to_id] += 1
        out_value[from_id] += val
        in_value[to_id] += val
        last_ts[from_id] = max(last_ts[from_id], ts)
        last_ts[to_id] = max(last_ts[to_id], ts)

    print(f'Done {class_name} {type_}: {len(edges)} edges, {len(address_to_id)} unique addresses.')
    return edges, out_count, in_count, out_value, in_value, last_ts, address_to_id, next_addr_id


def build_graph_from_addresses(all_edges, phish_set, address_to_id,
                               out_count, in_count, out_value, in_value, last_ts, save_path, value_scale=1e12):

    num_nodes = len(address_to_id)
    print(f'\nBuilding DGL graph with {num_nodes} nodes and {len(all_edges)} edges...')

    src_ids = [u for u, _ in all_edges]
    dst_ids = [v for _, v in all_edges]
    graph = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes)

    feat = torch.zeros((num_nodes, 5), dtype=torch.float32)
    for addr, idx in address_to_id.items():
        feat[idx, 0] = float(out_count.get(idx, 0))
        feat[idx, 1] = float(in_count.get(idx, 0))
        feat[idx, 2] = float(out_value.get(idx, 0)) / value_scale
        feat[idx, 3] = float(in_value.get(idx, 0)) / value_scale
        feat[idx, 4] = float(last_ts.get(idx, 0))

    labels = torch.zeros(num_nodes, dtype=torch.long)
    for addr, idx in address_to_id.items():
        if addr in phish_set:
            labels[idx] = 1
    print("Số lượng node phisher:", labels.sum().item())

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

    graph.ndata['feature'] = feat
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    dgl.save_graphs(save_path, [graph])

    print(f"Graph saved to {save_path}")
    return graph


if __name__ == "__main__":
    phish_set = load_phish_set('./raw/phisher_account.txt')
    address_to_id = {}
    next_addr_id = 0

    all_edges = []
    total_out = defaultdict(int)
    total_in = defaultdict(int)
    total_out_value = defaultdict(float)
    total_in_value = defaultdict(float)
    total_ts = defaultdict(int)

    for type_, cls in [('in', 'phisher'), ('out', 'phisher'),
                       ('in', 'normal'), ('out', 'normal')]:
        edges, out_c, in_c, out_v, in_v, ts, address_to_id, next_addr_id = process_address_graph(
            type_, cls, phish_set, address_to_id, next_addr_id
        )
        all_edges.extend(edges)
        for k, v in out_c.items(): total_out[k] += v
        for k, v in in_c.items(): total_in[k] += v
        for k, v in out_v.items(): total_out_value[k] += v
        for k, v in in_v.items(): total_in_value[k] += v
        for k, v in ts.items(): total_ts[k] = max(total_ts[k], v)

    build_graph_from_addresses(
        all_edges, phish_set, address_to_id,
        total_out, total_in, total_out_value, total_in_value, total_ts,
        './build_graph/b4e_address_graph_all'
    )
