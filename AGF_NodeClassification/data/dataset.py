import torch
from torch_geometric.datasets import Planetoid, DeezerEurope
import torch_geometric.transforms as T

def get_dataset(name, root='./data'):
    if name.lower() in ['cora', 'citeseer']:
        dataset = Planetoid(root=root, name=name, transform=T.NormalizeFeatures())
    elif name.lower() == 'deezer':
        dataset = DeezerEurope(root=root, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return dataset

def get_data(name, root='./data'):
    dataset = get_dataset(name, root)
    data = dataset[0]
    return dataset, data
