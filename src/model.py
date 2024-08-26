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

        self.input_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(-1).expand(-1, self.dim_in, self.dim_out)
        self.input_cache = x

        x_xnor = self.masks.tensor.logical_xor(x).logical_not()

        sums = x_xnor.sum(dim=1)

        x = sums > self.thresholds.tensor

        return x

    def backward(self, output_grad: torch.Tensor) -> torch.Tensor:

        # first, let's think about the gradient with respect to the threshold
        # if error is negative, then the threshold should be lowered (output was 0 when it should be 1)
        # grad = -1
        # if error is positive, then the threshold should be raised (output was 1 when it should be 0)
        # grad = 1
        # if error is zero, then no change is needed. grad = 0.

        # error has shape B x dim_out
        # thresholds has shape dim_out
        # cache[sums] has shape B x dim_out
        # cache[xnor] has shape B x dim_in x dim_out
        # cache[input] has shape B x dim_in

        B = output_grad.shape[0]

        self.thresholds.grad = -output_grad  # .float().mean(dim=0)

        # if a threshold grad is -1, that means we want 0s to flip to 1s in the activations
        # thus, bit-flip probability is not(activations)
        # if a threshold grad is +1, we want 1s to flip to 0s in the activations
        # thus, bit-flip probability is activations
        # if a threshold grad is zero, we want no bit flips to occur in the activations
        # thus, bit-flip probability is zeros.

        # because a bitflip in the mask always results in a bitflip in the activations,
        # the gradient of the masks is equivalent ot the gradient of the activations in this
        # case, I think.
        #
        # And the grad/error of the *input* is the same again! Because a bitflip in the
        # input results in a bitflip in the activations, too. Only this time, we assign a ternary
        # {-1, 0, 1} grad because this error is used to adjsut the thresholds in the next layer back. The
        # direction of the bitflip matters.

        self.masks.grad = torch.full((B, self.dim_in, self.dim_out), False)
        self.masks.grad = torch.where(
            self.thresholds.grad.unsqueeze(1) < 0.001,
            self.masks.tensor.logical_not(),
            self.masks.grad,
        )
        self.masks.grad = torch.where(
            self.thresholds.grad.unsqueeze(1) > 0.001,
            self.masks.tensor,
            self.masks.grad,
        )

        input_grad = torch.zeros(self.input_cache.shape, dtype=torch.int8)
        input_grad = torch.where(
            self.input_cache.logical_and(self.masks.grad), 1, input_grad
        )
        input_grad = torch.where(
            self.input_cache.logical_not().logical_and(self.masks.grad), -1, input_grad
        )

        self.thresholds.grad = self.thresholds.grad.float().mean(dim=0)  # dim_out
        self.masks.grad = self.masks.grad.float().mean(dim=0)  # dim_in x dim_out

        input_grad = input_grad.float().mean(dim=-1)  # B x dim_in

        return input_grad

    def parameters_iterator(self):
        return {"masks": self.masks, "thresholds": self.thresholds}

    def clamp(self):
        self.thresholds.tensor.clamp_(0, self.dim_in - 1)


class Model(nn.Module):
    def __init__(
        self, shape: tuple[int], hidden_dim: int, num_blocks: int, device: torch.device
    ):
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

        error = error.permute(1, 2, 0)

        for block in reversed(self.blocks):
            error = block.backward(error)

    def parameters_iterator(self):
        params = {}
        for i, block in enumerate(self.blocks):
            for name, tensor in block.parameters_iterator().items():
                params[f"block_{i}_{name}"] = tensor

        return params

    def clamp(self):
        for block in self.blocks:
            block.clamp()
