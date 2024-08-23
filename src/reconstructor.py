import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *


class Reconstructor(nn.Module):
    def __init__(self, shape, hidden_dim, num_blocks, device):
        super().__init__()

        self.shape = shape
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.pos_enc = get_binary_position_encoding(shape, device)

        self.pe_dim = self.pos_enc.shape[-1]

        self.hidden_dim = hidden_dim

        block_sizes = [self.pe_dim] + [self.hidden_dim * 2] * (num_blocks - 1) + [24]

        self.tree = GateTree(block_sizes)

    def forward(self, pos_enc) -> torch.Tensor:

        x = self.tree(pos_enc)

        return x.permute(2, 0, 1)

    def mutate(self, mutation_rate=0.01):
        self.tree.mutate(mutation_rate)



class GateTree(nn.Module):
    def __init__(self, block_sizes):
        super().__init__()
        self.blocks: list[GateBlock] = nn.Sequential()
        for dim_in, dim_out in zip(block_sizes[:-1], block_sizes[1:]):
            self.blocks.append(GateBlock(dim_in, dim_out))

    def forward(self, x):
        x = self.blocks(x)
        return x

    def mutate(self, mutation_rate):
        for block in self.blocks:
            block.mutate(mutation_rate)

class GateBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.
    def forward(self, x):
        return x
    def mutate(self, mutation_rate):
        pass