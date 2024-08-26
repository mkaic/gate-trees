from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from torchvision.io import write_jpeg
import torchvision.transforms.functional as TF
from tqdm import tqdm
from .reconstructor import *

from collections import deque

parser = ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0)
gpu = parser.parse_args().gpu
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

hidden_dim = 256
num_blocks = 3

iterations = 250_000
lr = 0.01
save = False

images_path = Path("recon/images")
images_path.mkdir(exist_ok=True, parents=True)
weights_path = Path("recon/weights")
weights_path.mkdir(exist_ok=True, parents=True)

# image_rgb = Image.open("jwst_cliffs.png").convert("RGB")
# image_rgb = Image.open("branos.jpg").convert("RGB")
image_rgb = Image.open("monalisa.jpg").convert("RGB")
# image_rgb = Image.open("minion.jpg").convert("RGB")
# image_rgb = Image.open("/workspace/projects/noah+brytan.png").convert("RGB")

image_rgb = TF.to_tensor(image_rgb).to(device)
image_rgb = TF.resize(image_rgb, 256)
image_rgb: torch.Tensor = (image_rgb * 255).to(torch.uint8)

write_jpeg(
    image_rgb.cpu(),
    "recon/original.jpg",
    quality=100,
)

c, h, w = image_rgb.shape

reconstructor = Reconstructor(
    shape=(h, w),
    hidden_dim=hidden_dim,
    num_blocks=num_blocks,
    device=device,
).to(device, dtype)

reconstructor = torch.compile(reconstructor)

# print(reconstructor)

num_params = sum([p.numel() for p in reconstructor.parameters()])

print(f"{num_params:,} bits | {num_params / 8 / 1024:.2f} KiB")

pbar = tqdm(range(iterations + 1))

accepted_mutations = deque(maxlen=1000)
improved_mutations = deque(maxlen=1000)

acceptance_rate = 1.0
improvement_rate = 1.0

best_output = None

with torch.no_grad():
    lowest_error = float("inf")
    for i in pbar:

        reconstructor.stage_mutation()

        output = reconstructor.forward()
        error = torch.mean(torch.abs((output - image_rgb).float()))

        if error <= lowest_error:

            reconstructor.accept_mutation()
            accepted_mutations.append(1)

            pbar.set_description(
                f"MAE: {lowest_error:.3f} | Acceptance: {acceptance_rate:.4f} | Improvement: {improvement_rate:.4f}"
            )

            if error < lowest_error:
                improved_mutations.append(1)
                lowest_error = error
                best_output = output.clone()

        else:
            reconstructor.revert_mutation()
            accepted_mutations.append(0)
            improved_mutations.append(0)

        acceptance_rate = sum(accepted_mutations) / len(accepted_mutations)
        improvement_rate = sum(improved_mutations) / len(improved_mutations)

        if i % 1000 == 0:
            to_save = best_output.cpu().to(torch.uint8)
            write_jpeg(
                to_save,
                f"recon/images/{i:04d}.jpg",
                quality=100,
            )
            write_jpeg(
                to_save,
                f"recon/latest.jpg",
                quality=100,
            )

    if save:
        torch.save(reconstructor.state_dict(), f"recon/weights/{i:04d}.pt")
