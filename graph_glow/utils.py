import os
import torch
import torch.nn as nn
import numpy as np

class Graph:
    def __init__(self, 
                 layout='locomotion',
                 scale=1):
        self.scale = scale
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, self.scale) 
        self.get_adjacency()

    def get_edge(self, layout):
        if layout == 'locomotion':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 0), (2, 1), (3, 2), (4, 3), 
                             (5, 0), (6, 5), (7, 6), (8, 7), 
                             (9, 0), (10, 9), (11, 10), (12, 11), 
                             (13, 11), (14, 13), (15, 14), (16, 15), 
                             (17, 11), (18, 17), (19, 18), (20, 19)]
            self.edge = self_link + neighbor_link
            self.center = 10
        else:
            raise ValueError('This layout is not supported!')
    
    def get_adjacency(self):
        valid_hop = range(0, self.scale+1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)
        
        A = []
        for hop in valid_hop:
            a_root = np.zeros((self.num_node, self.num_node))
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if self.hop_dis[j, i] == hop:
                        if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                            a_root[j, i] = normalize_adjacency[j, i]
                        elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                            a_close[j, i] = normalize_adjacency[j, i]
                        else:
                            a_further[j, i] = normalize_adjacency[j, i]
            if hop == 0:
                A.append(a_root)
            else:
                A.append(a_root + a_close)
                A.append(a_further)
        self.A = np.stack(A)
        
def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross3":
        return tensor[:, 0::3, ...], tensor[:, 1::3, ...], tensor[:, 2::3, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


def normalize_points_with_size(xy, width, height, flip=False):
    """Normalize scale points in image with size of image to (0-1).
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy[:, :, 0] /= width
    xy[:, :, 1] /= height
    if flip:
        xy[:, :, 0] = 1 - xy[:, :, 0]
    return xy


def scale_pose(xy):
    """Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        for j in range(xy.shape[3]):
            if min(xy_max[i,:,j] - xy_min[i,:,j]) > 0:
                xy[i,:,:,j] = ((xy[i,:,:,j] - xy_min[i,:,j]) / (xy_max[i,:,j] - xy_min[i,:,j])) * 2 - 1
    return xy.squeeze()


class LinearZeros(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.weight.data.zero_()
        self.bias.data.zero_()


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels, 
                 kernel_size: int=3, 
                 stride: int=1, 
                 padding='same',
                 logscale_factor=3):
        kernel_size = [kernel_size, kernel_size]
        stride = [stride, stride]
        pad_dict = {
            "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
            "valid": lambda kernel, stride: [0 for _ in kernel]
        }
        padding = pad_dict[padding](kernel_size, stride)
        
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(1, out_channels, 1))
        self.weight.data.zero_()
        self.bias.data.zero_()
    
    def forward(self, x):
        out = super().forward(x)
        return out * torch.exp(self.logs * self.logscale_factor)


    
class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, 
                              kernel_size=(1, 1), 
                              padding=(t_padding, 0), 
                              stride=(t_stride, 1), 
                              dilation=(t_dilation, 1),
                              bias=bias)
    
    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous()

class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0.0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = GraphConvolution(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (kernel_size[0], 1),
                                           (stride, 1),
                                           padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True)
                                 )
        
        if not residual:
            self.resudial = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels)
                                          )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x)


def linear_interpolation(z, step=10):
    z_interpo = torch.zeros_like(z)
    z_start = z[0:len(z):step]
    z_end = z[step-1:len(z):step]
    
    interpo_shape = [step-2, z.shape[1], z.shape[2]]
    
    for i in range(len(z_start)):
        z1 = z_start[i]
        z2 = z_end[i]
        c = torch.zeros(*interpo_shape)
        
        z_interpo[i*step] = z1
        z_interpo[(i+1)*step-1] = z2
        for s in range(step-2):
            c[s] = torch.lerp(z1, z2, s/(step-2))
            
        z_interpo[i*step+1:(i+1)*step-1] = c
    
    return z_interpo   
        # torch.zeros(step, z1.shape[1], z1.shape[2], )
        # for 
        