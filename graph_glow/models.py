import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
from .modules import ActNorm, InvertibleConv1x1, Permute, AffineCoupling
from .utils import LinearZeros, split_feature

class FlowStep(nn.Module):
    def __init__(self, num_in_channels, 
                 hidden_size,
                 actnorm_scale=1.0, 
                 flow_permutation='invconv', 
                 flow_coupling='affine', 
                 net_type='gcn',
                 graph_scale=1.0,
                 layout='locomotion',
                 LU_decomposed=True):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.LU_decomposed = LU_decomposed
        # ActNorm
        self.actnorm = ActNorm(num_in_channels, scale=actnorm_scale)
        # Permutation
        self.invconv = InvertibleConv1x1(num_in_channels, LU_decomposed=True)
        # Affine Coupling
        self.affine_coupling = AffineCoupling(in_channels=3, 
                                              hidden_size=hidden_size, 
                                              net_type=net_type,
                                              graph_scale=graph_scale,
                                              layout=layout,
                                              affine=True)

    def forward(self, x, logdet=None, reverse=False):
        if not reverse:
            out, logdet = self.actnorm(x, logdet)
            out, logdet = self.invconv(out, logdet)
            out, logdet = self.affine_coupling(out, logdet)
            return out, logdet
        else:
            out = self.affine_coupling(x, reverse=reverse)
            out = self.invconv(out, reverse=reverse)
            out = self.actnorm(out, reverse=reverse)
            return out


class Block(nn.Module):
    def __init__(self, in_channels, hidden_size, K, 
                 actnorm_scale=1.0, 
                 flow_permutation='invconv', 
                 flow_coupling='affine', 
                 net_type='gcn',
                 graph_scale=1.0,
                 layout='locomotion',
                 LU_decomposed=True):
        super().__init__()
        
        self.flows = nn.ModuleList()
        for i in range(K):
            self.flows.append(
                FlowStep(num_in_channels=in_channels,
                         hidden_size=hidden_size,
                         actnorm_scale=actnorm_scale,
                         flow_permutation='invconv', 
                         flow_coupling='affine', 
                         net_type='gcn',
                         graph_scale=graph_scale,
                         layout='locomotion',
                         LU_decomposed=True)
                )
        
            
    def forward(self, x, logdet=None, reverse=False):
        N, C, V = x.shape
        if not reverse:
            for flow in self.flows:
                x, logdet = flow(x, logdet)                   
                logp = self.logp(x)
                z = x
            return logdet, logp, z
        else:
            for flow in self.flows[::-1]:
                x = flow(x, reverse=reverse)
            return x
                
    @staticmethod
    def likelihood(x):
        device = x.device
        log2PI = torch.log(2 * torch.tensor(math.pi)).to(device)
        return -0.5 * (log2PI + x**2)

    @staticmethod
    def logp(x):
        likelihood = Block.likelihood(x)
        return torch.sum(likelihood, dim=[1, 2])
    

class Glow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.Glow.in_channels
        self.joints = cfg.Glow.joints
        self.block = Block(in_channels=self.in_channels,
                           hidden_size=cfg.Glow.hidden_channels,
                           K=cfg.Glow.K,
                           actnorm_scale=cfg.Glow.actnorm_scale,
                           flow_permutation=cfg.Glow.flow_permutation,
                           flow_coupling=cfg.Glow.flow_coupling,
                           net_type=cfg.Glow.net_type,
                           graph_scale=cfg.Glow.graph_scale,
                           layout=cfg.Glow.layout,
                           LU_decomposed=cfg.Glow.LU_decomposed)
    
    def generative_loss(self, nll):
        return torch.mean(nll)
    
    def negative_log_likelihood(self, logdet, logp):
        objective = logdet + logp
        logdet_factor = self.in_channels * self.joints
        nll = (-objective) / float(np.log(2.0) * logdet_factor)
        return nll
    
    def forward(self, x, reverse=None):
        if not reverse:
            logdet, logp, z = self.block(x)
            nll = self.negative_log_likelihood(logdet, logp)
            loss = self.generative_loss(nll)
            return z, loss
        else:
            with torch.no_grad():
                z = self.block(x, reverse=True)
            return z