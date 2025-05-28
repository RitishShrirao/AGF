import torch
import torch.nn as nn
from torch import Tensor
# from scipy.linalg import eigvals
# from torch.linalg import eigvals


class PolyConvFrame(nn.Module):
    '''
    A framework for polynomial graph signal filter.
    Args:
        conv_fn: the filter function, like PowerConv, LegendreConv,...
        depth (int): the order of polynomial.
        cached (bool): whether or not to cache the adjacency matrix. 
        alpha (float):  the parameter to initialize polynomial coefficients.
        fixed (bool): whether or not to fix to polynomial coefficients.
    '''
    def __init__(self,
                 conv_fn_type,
                 depth: int = 3,
                 cached: bool = True,
                 alpha: float = 1.0,
                 beta: float = 0.2,
                 fixI: bool = False):
        super().__init__()
        self.depth = depth
        self.basetheta = 1.0

        self.thetas = [nn.Parameter(torch.tensor(0.0), requires_grad=True) for i in range(self.depth+1)] 
        
        self.fixI = fixI
        if self.fixI:
            self.thetas[0] = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        
        self.thetas = nn.ParameterList(self.thetas)

        self.cached = cached
        self.adj = None
        self.H = []

        if conv_fn_type == 'monomial':
            self.conv_fn = self.PowerConv
        elif conv_fn_type == "legendre":
            self.conv_fn = self.JacobiConv
            self.alpha = self.beta = 0
        elif conv_fn_type == "chebyshev":
            self.conv_fn = self.JacobiConv
            self.alpha = self.beta = -0.5
        elif conv_fn_type == "jacobi":
            self.conv_fn = self.JacobiConv
            self.alpha = alpha
            self.beta = beta
    def forward(self, adj: Tensor):
        '''
        Args:
            x: node embeddings. of shape (number of nodes, node feature dimension)
            edge_index and edge_attr: If the adjacency is cached, they will be ignored.
        '''
        thetas = [self.basetheta * torch.tanh(i) for i in self.thetas]
        if self.fixI:
            thetas[0] = self.basetheta * self.thetas[0]
            
        xs = []
        theta_xs = []
        for L in range(self.depth+1):
            tx = self.conv_fn(L, xs, adj)
            xs.append(tx)
            theta_xs.append(thetas[L] * tx)
            
        out = sum(theta_xs)
        return out

    def PowerConv(self, L, xs, adj):
        '''
        Monomial bases.
        '''
        if L == 0: 
            return 1
        else:
            return (adj * xs[-1])


    def JacobiConv(self, L, xs, adj):
        if L == 0:
            return 1
        elif L == 1:
            return 0.5 * (self.alpha - self.beta + (self.alpha + self.beta + 2) * adj)
        else:
            A_l = ((2*L+self.alpha+self.beta) * (2*L+self.alpha+self.beta-1)) / (2*L*(L+self.alpha+self.beta))
            B_l = ((2*L+self.alpha+self.beta-1) * (self.alpha**2-self.beta**2)) / (2*L*(L+self.alpha+self.beta)*(2*L+self.alpha+self.beta-2))
            C_l = ((L+self.alpha-1)*(L+self.beta-1)*(2*L+self.alpha+self.beta)) / (L*(L+self.alpha+self.beta)*(2*L+self.alpha+self.beta-2))
            
            return (A_l * adj + B_l) * xs[-1] - C_l * xs[-2]