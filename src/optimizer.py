import torch
from .model import GradTensor


class StochasticUpdatingSGD:
    def __init__(self, parameters: dict[str, GradTensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for name, param in self.parameters.items():
            device = param.tensor.device

            if "threshold" in name:
                update_mask = torch.rand(param.tensor.shape, device=device) < (
                    param.grad * self.lr
                )
                updates = torch.sign(param.grad) * update_mask
                param.tensor = param.tensor + updates
            elif "mask" in name:
                update_mask = torch.rand(param.tensor.shape, device=device) < (
                    param.grad.abs() * self.lr
                )
                param.tensor = param.tensor.logical_xor(
                    update_mask
                )  # flips param bits where mask is true

            param.grad = None
