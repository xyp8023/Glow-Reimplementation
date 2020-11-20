# Here we show one demo how to load model and use umap to visualize the latent space

from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, utils
from ..dataset import foldingshirt, sss


from ..models.model import Glow
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import umap
import seaborn as sns


parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def umapvis(zout_list, label_list):
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for i in range(len(zout_list)):
        plt.figure(i)
        sns.set(style='white', rc={'figure.figsize': (10, 8)})
        # standard_embedding = umap.UMAP(random_state=42).fit_transform(lat[0][:100,:])
        # scat = plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=lat[1][:100], s=20, cmap='Spectral')
        standard_embedding = umap.UMAP(random_state=42, n_neighbors=30).fit_transform(zout_list[i])
        scat = plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=label_list, s=20, cmap='Spectral')
        plt.title("z_out "+str(i))
        plt.legend(handles=scat.legend_elements()[0], labels=classes)
        # plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=0.1, cmap='Spectral');
        plt.show()


def latDataset(model=None, n_channel=None, input_size=None, n_flows=None, n_blocks=None,
               vis_loader=None, device_vis=None, uselabel=False):
    # get number of vis points
    numdata = len(vis_loader.dataset)

    # get zshape
    z_shapes = calc_z_shapes(n_channel, input_size, n_flows, n_blocks)

    # construct an empty container for z and labels
    zout_list = []
    for i in range(n_blocks):
        zout_list.append(torch.zeros(numdata, z_shapes[i][0] * z_shapes[i][1] * z_shapes[i][2]))

    label_list = torch.zeros(numdata)

    # get the z and labels
    start = 0
    for batch_idx, data_in in enumerate(vis_loader):

        if uselabel is True:
            x_in = data_in[0]

        else:
            x_in = data_in


        
        x_in = x_in.to(device_vis)
        numdata = x_in.shape[0]
        with torch.no_grad():
        
            _, _, zout = model(x_in)
    
        for i in range(n_blocks):
            zout_list[i][start: start + numdata] = zout[i].view(numdata, -1).detach()  # torch.flatten(zout[i], start_dim=1,end_dim=3)

        if uselabel is True:
            x_class = data_in[1]
            label_list[start: start + numdata] = x_class.detach()
        else:
            label_list[start: start + numdata] = 0

        start = start + numdata
        print(batch_idx)

    return zout_list, label_list


if __name__ == "__main__":
    # load model

    n_block = 3
    n_flow = 32  
    n_bits = 8
    n_channels = 3
    image_size = 32
    batchsize = 256
    vislabel = True


    n_bins = 2.0 ** n_bits

    model_single = Glow(n_channels, n_blocks=n_block, n_flows=n_flow).to(device)
    model = nn.DataParallel(model_single)


    
    # CIFAR10 model checkpoint
    model_path = sys.argv[1] 

    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    # load data
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            #             transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    # STEP 1
    # define the dataset here
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=256, num_workers=16)

    # STEP 2
    zout_list, label_list = latDataset(model=model, n_channel=n_channels, input_size=image_size, n_flows=n_flow, n_blocks=n_block,
               vis_loader=dataloader, device_vis=device, uselabel=vislabel)

    # STEP 3
    umapvis(zout_list, label_list)