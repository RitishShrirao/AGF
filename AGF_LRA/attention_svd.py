
import torch
import torch.nn as nn
import math

import torch.nn.functional as F
from polyconv import PolyConvFrame

class SVDAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.poly = PolyConvFrame(config["poly_type"], depth=config["K"], alpha=config["alpha"], beta=config["beta"], fixI=config["fixI"])
        self.reg = config["reg"]

    def forward(self, U, Sigma, svd_V, V, mask):
        Q = nn.functional.softmax(U, dim=-1) 
        K = nn.functional.softmax(svd_V - 1e9 * (1 -  mask[:, None, :, None]), dim=-2) 

        Sigma = nn.functional.sigmoid(Sigma)
        graph_filter = self.poly(Sigma)

        X = torch.matmul(Q * graph_filter, torch.matmul(torch.transpose(K, -2, -1), V))
        # X = torch.matmul(torch.matmul(Q, graph_filter), torch.matmul(torch.transpose(K, -2, -1), V))
        sym = torch.matmul(Q.transpose(-2,-1), Q)
        sym -= torch.eye(sym.shape[-1]).to(sym.device)
        ortho_loss = self.reg * sym.abs().mean(dim=[1,2,3])

        sym = torch.matmul(K.transpose(-2,-1), K)
        sym -= torch.eye(sym.shape[-1]).to(sym.device)
        ortho_loss += self.reg * sym.abs().mean(dim=[1,2,3])
        return X, ortho_loss