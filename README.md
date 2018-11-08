# PRUNE-pytorch
A pytorch implementation of [PRUNE : Preserving Proximity and Global Ranking for Network Embedding](https://nips.cc/Conferences/2017/Schedule?showEvent=9301)

---
## Desciprtion

**PRUNE** is an unsupervised generative approach for network embedding.

Design properties **PRUNE** satisfies: scalability, asymmetry, unity and simplicity.

The approach entails a multi-task Siamese neural network to connect embeddings and our objective, preserving global node ranking and local proximity of nodes.

Deeper analysis for the proposed architecture and objective can be found in the paper (please see - *[PRUNE](https://nips.cc/Conferences/2017/Schedule?showEvent=9301)*) <br>

## Requirement
- pytorch==0.4.0
- python==3.5.2

## Usage
#### Clone the repository
```
git clone https://github.com/j40903272/PRUNE-pytorch
```
#### Prepare data
Prepare a graph with edge lists in **<from_node**, **to_node>**  format

Examples in : [edgelist.txt](https://github.com/j40903272/PRUNE-pytorch/blob/master/example/edgelist.txt)
#### Training
```
python3 train.py ../example/edgelist.txt
```
The PRUNE model would be stored in **prune.pt**

The embedding weights would be in **prune_weight.pkl**

## Example
```
import numpy as np
from preprocess import preprocess
from model import PRUNE
from train import train

graph = np.loadtxt(args.input_graph).astype(np.int64)
nodeCount = int(graph.max()) + 1
data_loader, PMI_dict = preprocess(graph)

model = PRUNE(nodeCount).cuda()
train(model, data_loader)

emb_weight = model.node_emb.weight.data.cpu().numpy()
with open('prune_weight.pkl', 'wb') as f:
    pickle.dump(emb_weight, f)
```
