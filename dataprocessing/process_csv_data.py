import argparse
from tqdm import tqdm
from pathlib import Path

import dgl
from dgl.data.utils import save_graphs
import numpy as np
import pandas as pd
import torch


class Reindex:
    def __init__(self, args):
        self.name = args.data
        self.user_idx = {}
        self.item_idx = {}
        self.curr_idx = 0

    def reindex_bipartite(self, usr_id, item_id):
        if usr_id in self.user_idx.keys():
            usr_id = self.user_idx[usr_id]
        else:
            self.user_idx[usr_id] = self.curr_idx
            usr_id = self.curr_idx
            self.curr_idx += 1

        if item_id in self.user_idx.keys():
            item_id = self.user_idx[item_id]
        else:
            self.user_idx[item_id] = self.curr_idx
            item_id = self.curr_idx
            self.curr_idx += 1

        return usr_id, item_id


def process_data(args):
    if not (args.data == 'wikipedia' or args.data == 'reddit'):
        print('Please check the dataset name')

    Path('data/{}/'.format(args.data)).mkdir(parents=True, exist_ok=True)
    CSV_PATH = 'data/{}.csv'.format(args.data)
    nodes_feat_dim = 172

    reidx = Reindex(args)
    u_list, i_list, ts_list, label_list, idx_list, feat_l = [], [], [], [], [], []
    with open(CSV_PATH) as data_csv:
        for idx, line in enumerate(tqdm(data_csv)):
            if idx == 0:
                continue
            e = line.strip().split(',')
            u, i = int(e[0]), int(e[1])
            u, i = reidx.reindex_bipartite(u, i)
            ts = float(e[2])
            label = float(e[3])
            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            feat_l.append(feat)

    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l), np.zeros((reidx.curr_idx, nodes_feat_dim))


def create_Dgl_Graph(graph_df, edges_features, nodes_features):
    src = torch.tensor(graph_df.u.values)
    dst = torch.tensor(graph_df.i.values)
    label = torch.tensor(graph_df.label.values, dtype=torch.float32)

    timestamp = torch.tensor(graph_df.ts.values, dtype=torch.float32)

    edge_feat = torch.tensor(edges_features, dtype=torch.float32)
    node_feat = torch.tensor(nodes_features, dtype=torch.float32)

    g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])))
    g.edata['label'] = label.repeat(2).squeeze()
    g.edata['timestamp'] = timestamp.repeat(2).squeeze()
    g.edata['feat'] = edge_feat.repeat(2, 1).squeeze()
    g.ndata['feat'] = node_feat.squeeze()
    print(g)
    return g


parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')

args = parser.parse_args()
graph_df, edges_features, nodes_features = process_data(args)

GRAPH_OUT = 'data/{}/graph.csv'.format(args.data)
EDGES_OUT = 'data/{}/edges.npy'.format(args.data)
NODES_OUT = 'data/{}/nodes.npy'.format(args.data)

graph_df.to_csv(GRAPH_OUT, index_label='idx')
np.save(EDGES_OUT, edges_features)
np.save(NODES_OUT, nodes_features)

dgl_graph = create_Dgl_Graph(graph_df, edges_features, nodes_features)

save_graphs('data/{}/graph.bin'.format(args.data), dgl_graph)
