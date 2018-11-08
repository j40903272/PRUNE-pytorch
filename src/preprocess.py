import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def preprocess(graph, alpha=5.0):
    nodeCount = int(graph.max()) + 1
    out_degrees = np.zeros(nodeCount)
    in_degrees = np.zeros(nodeCount)
    for node_i, node_j in graph:
        out_degrees[node_i] += 1
        in_degrees[node_j] += 1
    # avoid divied zero
    out_degrees[out_degrees == 0] = 1
    in_degrees[in_degrees == 0] = 1
    
    
    PMI_dict = {}
    PMI = np.zeros((len(graph))).astype('float32')
    for idx, edge in enumerate(graph):
        fromId, toId = edge
        pmi = len(graph) / alpha / out_degrees[fromId] / in_degrees[toId]
        PMI[idx] = np.log(pmi)
        PMI_dict[(fromId, toId)] = np.log(pmi)
    PMI[PMI < 0] = 0
    
    head_node = graph[:, 0]
    tail_node = graph[:, 1]

    train_loader = DataLoader(
        dataset=TensorDataset(
            torch.from_numpy(head_node.astype('int64')),
            torch.from_numpy(tail_node.astype('int64')),
            torch.from_numpy(PMI.astype('float32')),
            torch.from_numpy(in_degrees[head_node].astype('float32')),
            torch.from_numpy(out_degrees[tail_node].astype('float32'))
        ),
        batch_size=1024,
        shuffle=True,
        num_workers=8)
    
    return train_loader, PMI_dict