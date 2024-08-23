from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from torchvision.io import write_jpeg
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from .reconstructor import *
import kornia.color as kc
import torch.nn as nn

parser = ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0)
gpu = parser.parse_args().gpu
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

hidden_dim = 24
num_layers = 4

iterations = 2000
lr = 0.01
save = False

images_path = Path("recon/images")
images_path.mkdir(exist_ok=True, parents=True)
weights_path = Path("recon/weights")
weights_path.mkdir(exist_ok=True, parents=True)

image_rgb = Image.open("jwst_cliffs.png").convert("RGB")
# image_rgb = Image.open("branos.jpg").convert("RGB")
# image_rgb = Image.open("monalisa.jpg").convert("RGB")
# image_rgb = Image.open("minion.jpg").convert("RGB")
# image_rgb = Image.open("/workspace/projects/noah+brytan.png").convert("RGB")

image_rgb = to_tensor(image_rgb).to(device)
image_rgb = (image_rgb * 255).to(torch.uint8)

write_jpeg(
    image_rgb.cpu(),
    "recon/original.jpg",
    quality=100,
)

c, h, w = image_rgb.shape

reconstructor = Reconstructor(
    shape=(h, w),
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    device=device,
).to(device, dtype)


num_params = sum([p.numel() for p in reconstructor.parameters()])

print(f"{num_params:,} trainable parameters")
print(f"{num_params / 1024 / 8:.2f} kB")

pbar = tqdm(range(iterations + 1))

best_loss = float("inf")
for i in pbar:
    output_bits = reconstructor()

    error_bits = output_bits.bitwise_xor(image_bits)

    pbar.set_description(f"Error: {error_rgb}")

    if i % 100 == 0:
        output_rgb = output_rgb.cpu()
        write_jpeg(
            output_rgb,
            f"recon/images/{i:04d}.jpg",
            quality=100,
        )
        write_jpeg(
            output_rgb,
            f"recon/latest.jpg",
            quality=100,
        )

if save:
    torch.save(reconstructor.state_dict(), f"recon/weights/{i:04d}.pt")
