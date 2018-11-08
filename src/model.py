import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class PRUNE(nn.Module):
    def __init__(self, nodeCount, n_latent=128, n_emb=128, n_prox=64):
        super(PRUNE, self).__init__()
        '''
        Parameters
        ----------
        n_latent : hidden layer dimension
        n_emb    : node embedding dimension
        n_prox   : proximity representation dimension
        '''
        
        # Embedding
        self.node_emb = nn.Embedding(nodeCount, n_emb)
        
        # W_shared
        w_init = np.identity(n_prox) + abs(np.random.randn(n_prox, n_prox) / 1000.0)
        self.w_shared = torch.from_numpy(w_init).float()
        if torch.cuda.is_available():
            self.w_shared = self.w_shared.cuda()
        
        # global node ranking score
        self.rank = nn.Sequential(
            self.node_emb,
            nn.Linear(n_latent, n_latent),
            nn.ELU(),
            nn.Linear(n_latent, 1),
            nn.Softplus()
        )
        
        # proximity representation
        self.prox = nn.Sequential(
            self.node_emb,
            nn.Linear(n_latent, n_latent),
            nn.ELU(),
            nn.Linear(n_latent, n_prox),
            nn.ReLU()
        )
        self.init_weight()
        
        
    def init_weight(self):
        torch.nn.init.xavier_normal_(self.node_emb.weight)
        torch.nn.init.xavier_normal_(self.w_shared)
        for layer in self.rank:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
        for layer in self.prox:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
    
    
    def forward(self, head, tail, pmi, indeg, outdeg, lamb=0.01):
        head_rank = self.rank(head)
        head_prox = self.prox(head)
        tail_rank = self.rank(tail)
        tail_prox = self.prox(tail)
        
        # preserving proximity
        w = F.relu(self.w_shared)
        zWz = (head_prox * torch.matmul(tail_prox, w)).sum(1)
        prox_loss = ((zWz - pmi)**2).mean()
        
        # preserving global ranking
        rank_loss = indeg * (-tail_rank / indeg + head_rank / outdeg).pow(2)
        rank_loss = rank_loss.mean()
        
        total_loss = prox_loss + lamb * rank_loss
        return total_loss
