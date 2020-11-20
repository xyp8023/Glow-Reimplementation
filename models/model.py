import torch
from torch import nn
from .module import MultiScaleBlock


class Glow(nn.Module):
    def __init__(self, in_channel, n_blocks, n_flows):
        super().__init__()
        # set hyper-parameters
        self.in_channel = in_channel
        self.n_blocks = n_blocks # l
        self.n_flows = n_flows # K
        # self.permutation = permutation
        
        # initial the blocks list
        self.blocks = nn.ModuleList()
        for i in range(n_blocks - 1):
            single_block = MultiScaleBlock(self.in_channel, self.n_flows, split=True)
            self.blocks.append(single_block)
            self.in_channel = self.in_channel * 2
        last_block = MultiScaleBlock(self.in_channel, self.n_flows, split=False)
        self.blocks.append(last_block)
        
    def forward(self, input):
        log_p_sum = 0
        logdet = None
        out = input
        z_outs = []

        for block in self.blocks:
            # out, logdet_, log_p, z_new = block(out)
            out, logdet, log_p, z_new = block(out, logdet)
            z_outs.append(z_new)
            # logdet = logdet + logdet_

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    
    def reverse(self, output):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(output[-1], output[-1])

            else:
                # input = block.reverse(input, output[-(i + 1)])
                input = block.reverse(input, output[-(i + 1)])

        return input

