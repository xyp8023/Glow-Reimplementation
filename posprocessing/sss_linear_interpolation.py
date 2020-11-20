# Here we show one demo how to load model, do linear interpolation in latent space and generate samples

from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse
from dataset.foldingshirt import shirtDataset
from dataset.sss import sssDataset
import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from utils.utils import linearInterpolate
from models.model import Glow
import sys 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
#             transforms.CenterCrop(image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


    dataset = sssDataset(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes





def run(model):
    dataset = iter(sample_data(data_path, batch, img_size))
    n_bins = 2.0 ** n_bits

    with tqdm(range(iteration)) as pbar:
        for i in pbar:
            image= next(dataset)

            image = image * 255

            if n_bits < 8:
                image = torch.floor(image / 2 ** (8 - n_bits))

            image = image / n_bins - 0.5

            image = image.to(torch.device('cuda'))

            log_p, logdet, z_sample = model.module(image)
            output = linearInterpolate(model_single, image[0].unsqueeze(0), image[1].unsqueeze(0), interpo_num=interpo_num)

            utils.save_image(
                        output.cpu().data,
                        f"outputs/sss_{str(nnn)}_linear_{str(interpo_num)}_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=interpo_num+1,
                        range=(-0.5, 0.5),
                    )
            utils.save_image(
                    image.cpu().data,
                    f"outputs/sss_{str(nnn)}_{str(i + 1).zfill(6)}.png",
                    normalize=True,
                    nrow=interpo_num+1,
                    range=(-0.5, 0.5),
                )


# SSS model checkpoint
model_path = sys.argv[1] 

# sss data path
data_path = sys.argv[2] 

batch = 2
img_size = 64
n_flow = 48
n_block = 4
n_bits = 8
n_channel = 1

n_sample = 10
iteration = 1
interpo_num=9

with torch.no_grad():
    model_single = Glow(
        n_channel, n_block, n_flow
    )


    model = nn.DataParallel(model_single)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model = model.to(device)

    
    for nnn in range(20):

        run(model)
