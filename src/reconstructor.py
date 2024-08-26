import torch
import torch.nn as nn
from .utils import *
from icecream import ic
import random

class Reconstructor(nn.Module):
    def __init__(self, shape, hidden_dim, num_blocks, device):
        super().__init__()

        self.shape = shape
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.pos_enc = get_binary_position_encoding(shape, device)

        self.pe_dim = self.pos_enc.shape[-1]

        self.hidden_dim = hidden_dim

        block_sizes = [self.pe_dim] + [self.hidden_dim] * (num_blocks - 1) + [24]

        self.tree = GateTree(block_sizes)

    def forward(self) -> torch.Tensor:

        x = self.tree.forward(self.pos_enc)

        return x.permute(2, 0, 1)

    def stage_mutation(self):
        self.tree.stage_mutation()

    def revert_mutation(self):
        self.tree.revert_mutation()

    def accept_mutation(self):
        self.tree.accept_mutation()

    def backward(self)


class GateTree(nn.Module):
    def __init__(self, block_sizes: list[int]):
        super().__init__()
        self.blocks: list[GateBlock] = nn.Sequential()
        for dim_in, dim_out in zip(block_sizes[:-1], block_sizes[1:]):
            self.blocks.append(GateBlock(dim_in, dim_out))

        self.mutated_block_idx = None
        self.mutated_parameter_idx = None
        self.original_value = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, shape[-1])
        x = self.blocks(x)
        x = x.view(*shape[:-1], 3, 8)
        x = bits_to_int(x)
        return x

    def stage_mutation(self):
        self.mutated_block = self.blocks[random.randint(0, len(self.blocks) - 1)]
        self.mutated_block.stage_mutation()

    def revert_mutation(self):
        self.mutated_block.revert_mutation()

    def accept_mutation(self):
        self.mutated_block.accept_mutation()


class GateBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.thresholds = nn.Parameter(
            torch.randint(
                0,
                self.dim_in,
                (self.dim_out,),
            ),
            requires_grad=False,
        )
        self.masks = nn.Parameter(
            torch.rand(self.dim_in, self.dim_out) < 0.5, requires_grad=False
        )

        self.chunk_size = self.dim_out // 2

        self.mutation_cache = None
        self.mutation_idx = None

        self.input_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        self.input_cache = x

        x = x.unsqueeze(-1).expand(-1, self.dim_in, self.dim_out)

        x_xnor = self.masks.logical_xor(x).logical_not()

        sums = x_xnor.sum(dim=1)

        x = sums > self.thresholds

        return x
    
    def stage_mutation(self):
        self.mutation_idx = random.randint(0, self.dim_out - 1)
        self.mutation_cache = self.thresholds[self.mutation_idx].clone()

        self.thresholds[self.mutation_idx] = random.randint(0, self.dim_in - 1)

    def revert_mutation(self):
        self.thresholds[self.mutation_idx] = self.mutation_cache

        self.mutation_cache = None
        self.mutation_idx = None

    def accept_mutation(self):
        self.mutation_cache = None
        self.mutation_idx = None
