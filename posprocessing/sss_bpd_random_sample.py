# Here we show one demo how to load model, calulate bpd, random sample from latent space and generate samples

from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from ..models.model import Glow
from ..dataset.sss import sssDataset 
from ..dataset.foldingshirt import shirtDataset_test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import torchvision
import sys 

def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes

def calc_loss(log_p, logdet, image_size, n_bins, n_channel):
    
    n_pixel = image_size * image_size * n_channel

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )

# load SSS model
n_block = 4
n_channel = 1
n_flow = 48
model_single= Glow(1, n_blocks=n_block, n_flows=n_flow)
model = nn.DataParallel(model_single) 
image_size = 64
n_bits = 8 

# SSS model checkpoint
model_path = sys.argv[1] 


model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

# sss test data path
data_path = sys.argv[2] 


transform = transforms.Compose(
        [
            transforms.Resize(image_size),
#             transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

dataset = sssDataset(data_path, transform=transform)
loader = DataLoader(dataset, shuffle=True, batch_size=16, num_workers=16)



loader = iter(loader)
bpd_all = 0
for i, image in enumerate(loader):
    with torch.no_grad():

        
        n_bins = 2.0 ** n_bits
        image = image.to(device)

        image = image * 255

        if n_bits < 8:
            image = torch.floor(image / 2 ** (8 - n_bits))

        image = image / n_bins - 0.5
        log_p, logdet, _ = model(image)
        bpd, _, _ = calc_loss(log_p, logdet, image_size, n_bins, n_channel)

        bpd_all = bpd_all + bpd.mean()

print("bpd is: ", bpd_all/(i+1))



# random sample
temp_list = [0.1, 0.6, 0.7,0.8, 0.9,1.0, 1.1, 1.2]

n_sample = 8
fig, axs = plt.subplots(len(temp_list),n_sample)

for row, temp in enumerate(temp_list):
    
    z_sample = []
    z_shapes = calc_z_shapes(n_channel, image_size, n_flow, n_block)
    for i in range(n_block):
        z = z_shapes[i]
        z_sample.append((torch.randn(n_sample, *z) * temp).to(device))


    with torch.no_grad():
        image_hat = model_single.reverse(z_sample)

    for i in range(n_sample):
        sss_hat_array = image_hat[i].squeeze().cpu().detach().numpy()
        sss_hat_array = (sss_hat_array-sss_hat_array.min())/(sss_hat_array.max()-sss_hat_array.min())
        sss_hat_array = np.uint8(sss_hat_array*255.)

        axs[row][i].imshow(sss_hat_array, cmap="gray")
        axs[row][i].set_xticks([]) 
        axs[row][i].set_yticks([]) 
plt.savefig("./random_sample_sss_"+str(temp_list)+".png", dpi=200)