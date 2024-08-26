import torch
import torch.nn as nn
from icecream import ic

from .utils import bits_to_int, get_binary_position_encoding


class GradTensor:
    def __init__(self, tensor: torch.Tensor, grad: torch.Tensor = None):
        self.tensor = tensor
        self.grad = grad if grad is not None else torch.empty_like(tensor)

    def to(self, device: torch.device) -> "GradTensor":
        self.tensor = self.tensor.to(device)
        self.grad = self.grad.to(device)
        return self


class Block(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.thresholds = GradTensor(torch.randint(0, self.dim_in, (self.dim_out,)))
        self.masks = GradTensor(torch.rand(self.dim_in, self.dim_out) < 0.5)

        self.cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.cache["input"] = x

        x = x.unsqueeze(-1).expand(-1, self.dim_in, self.dim_out)

        x_xnor = self.masks.tensor.logical_xor(x).logical_not()
        self.cache["xnor"] = x_xnor

        sums = x_xnor.sum(dim=1)
        self.cache["sums"] = sums

        x = sums > self.thresholds.tensor
        self.cache["output"] = x

        return x


class Model(nn.Module):
    def __init__(self, shape: tuple[int], hidden_dim: int, num_blocks: int, device: torch.device):
        super().__init__()

        self.pos_enc = get_binary_position_encoding(shape, device)

        block_sizes = [self.pos_enc.shape[-1]] + [hidden_dim] * (num_blocks - 1) + [24]

        self.blocks: list[Block] = nn.ModuleList()
        for dim_in, dim_out in zip(block_sizes[:-1], block_sizes[1:]):
            self.blocks.append(Block(dim_in, dim_out))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        shape = self.pos_enc.shape

        x = self.pos_enc.view(-1, shape[-1])

        for block in self.blocks:
            x = block.forward(x)

        x = x.view(*shape[:-1], 3, 8)

        x = bits_to_int(x)

        x = x.permute(2, 0, 1)

        error = x - image

        return x, error

    def backward(self, error: torch.Tensor):
        pass
