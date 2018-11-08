import pickle
import torch
import numpy as np
import argparse

from preprocess import preprocess
from model import PRUNE

parser = argparse.ArgumentParser(description='Malconv-keras classifier training')
parser.add_argument('input_graph', type=str, default='../example/edgelist.txt')


def train(model, train_loader, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        print('Epoch:', epoch)
        total_loss = []
        for head, tail, pmi, indegr, outdegr in train_loader:
            if torch.cuda.is_available():
                head, tail, pmi, indegr, outdegr = head.cuda(), tail.cuda(), pmi.cuda(), indegr.cuda(), outdegr.cuda()
            loss = model(head, tail, pmi, indegr, outdegr)
            total_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('train_loss:{:.4f}'.format(np.mean(total_loss)))
    torch.save({'state_dict': model.state_dict()}, 'prune.pt')

    
if __name__ == '__main__':
    args = parser.parse_args()
    graph = np.loadtxt(args.input_graph).astype(np.int64)
    nodeCount = int(graph.max()) + 1
    data_loader, PMI_dict = preprocess(graph)
    
    model = PRUNE(nodeCount)
    if torch.cuda.is_available():
        model.cuda()
    train(model, data_loader)
    
    emb_weight = model.node_emb.weight.data.cpu().numpy()
    with open('prune_weight.pkl', 'wb') as f:
        pickle.dump(emb_weight, f)
    with open('pmi.pkl', 'wb') as f:
        pickle.dump(PMI_dict, f)
