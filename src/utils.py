import torch
import math
from icecream import ic


def ints_to_bits(ints: torch.Tensor, num_bits: int) -> torch.Tensor:
    # snippet to convert from int to binary is adapted from https://stackoverflow.com/questions/62828620/how-do-i-convert-int8-into-its-binary-representation-in-pytorch
    mask = 2 ** torch.arange(num_bits - 1, -1, -1, dtype=torch.int, device=ints.device)
    bits = ints.unsqueeze(-1).bitwise_and(mask).ne(0)
    return bits


def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    num_bits = bits.shape[-1]
    bits = bits.int()
    scales = 2 ** torch.arange(
        num_bits - 1, -1, -1, device=bits.device, dtype=torch.int
    )
    return (bits * scales).sum(dim=-1)


def get_binary_position_encoding(shape, device):

    longest_side = max(shape)
    num_frequencies = math.ceil(math.log2(longest_side))

    positions = torch.stack(
        torch.meshgrid(
            *[torch.arange(i, dtype=torch.int, device=device) for i in shape],
            indexing="ij"
        ),
        dim=-1,
    )

    positions = ints_to_bits(positions, num_bits=num_frequencies)

    positions = positions.view(*shape, -1)

    return positions
