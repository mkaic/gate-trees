import shutil
from argparse import ArgumentParser
from collections import deque
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.io import write_jpeg
from tqdm import tqdm
from icecream import ic

from .model import Model
from .optimizer import StochasticUpdatingSGD

parser = ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0)
gpu = parser.parse_args().gpu
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

hidden_dim = 128
num_blocks = 4

iterations = 250_000
lr = 0.01
save = False

target_image_path = Path("monalisa.jpg")
image = Image.open(target_image_path).convert("RGB")

images_path = Path("recon/images")
images_path.mkdir(exist_ok=True, parents=True)
weights_path = Path("recon/weights")
weights_path.mkdir(exist_ok=True, parents=True)
shutil.copy(target_image_path, "recon/original.jpg")

image = TF.to_tensor(image).to(device)
image = TF.resize(image, 256)
image: torch.Tensor = (image * 255).to(torch.uint8)

reconstructor = Model(
    image,
    hidden_dim=hidden_dim,
    num_blocks=num_blocks,
    device=device,
).to(device)

# reconstructor: Model = torch.compile(reconstructor)

optimizer = StochasticUpdatingSGD(reconstructor.parameters_iterator(), lr=lr)

num_params = sum(
    [p.tensor.numel() for p in reconstructor.parameters_iterator().values()]
)
print(f"{num_params:,} bits | {num_params / 8 / 1024:.2f} KiB")

pbar = tqdm(range(iterations + 1))
mae_deque = deque(maxlen=100)

with torch.no_grad():
    for i in pbar:

        output, pixel_mae = reconstructor.forward(image)
        reconstructor.backward()
        optimizer.step()

        mae_deque.append(pixel_mae)

        pbar.set_description(
            f"MAE (1): {pixel_mae.item():.4f} | MAE (100): {sum(mae_deque) / len(mae_deque):.4f}"
        )

        if i % 100 == 0:
            to_save = output.cpu().to(torch.uint8)
            write_jpeg(
                to_save,
                f"recon/images/{i:06d}.jpg",
                quality=100,
            )
            write_jpeg(
                to_save,
                f"recon/latest.jpg",
                quality=100,
            )

    if save:
        torch.save(reconstructor.state_dict(), f"recon/weights/{i:04d}.pt")
