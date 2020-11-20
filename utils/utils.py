from math import log
import torch
from dataset.sss import sssDataset
from dataset.foldingshirt import shirtDataset_train
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_data(args):
    if args.model == "cifar10":
        transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    elif args.model == "celeba":
        transform = transforms.Compose(
            [
                transforms.CenterCrop(args.image_size*2), 
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset = datasets.ImageFolder(args.path, transform=transform)

    elif args.model == "shirt" :
        transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset = shirtDataset_train(args.path, transform=transform)

    elif args.model == "sss":
        transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset = sssDataset(args.path, transform=transform)

    else:
        print("Oops!  That model was not implemented!")
        raise(ValueError)
        

    loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=args.batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def generative_loss(logp, logdet, img_size, n_bits=8):
    a = 2 ** n_bits
    num_pixels = img_size * img_size * 3
    
    # loss = -log(a) * num_pixels + logp.mean() + logdet.mean()
    loss = -log(a) * num_pixels + logp.mean() + logdet.mean()

    # loss = (-loss / (log(2) * num_pixels)).sum()
    # logp = (logp / (log(2) * num_pixels)).sum()
    # logdet = (logdet / (log(2) * num_pixels)).sum()
    
    loss = (-loss / (log(2) * num_pixels))
    logp = (logp.mean() / (log(2) * num_pixels))
    logdet = (logdet.mean() / (log(2) * num_pixels))
    # print(logdet)
    return loss, logp, logdet

def random_sample(N, C=3, H=32, W=32, n_blocks=3, n_flows=32, temp=0.8):
    z_outs_random_sample = []
    for i_b in range(n_blocks-1):
        z_outs_random_sample.append(temp*torch.randn(N, C*2**(i_b+1), H//2**(i_b+1), W//2**(i_b+1)).to(device))


    z_outs_random_sample.append(temp*torch.randn(N,C*2**(n_blocks+1),H//2**(n_blocks),W//2**(n_blocks)).to(device))

    return z_outs_random_sample

def linearInterpolate(glowmodel, img1, img2, interpo_num=39):
    _, _, z1 = glowmodel(img1)
    _, _, z2 = glowmodel(img2)
    zspace = []
    for i in range(len(z1)):
        single_interpo = torch.zeros(interpo_num+1, z1[i].shape[1], z1[i].shape[2], z1[i].shape[3]).to(device)
        single_interpo[0, :] = z1[i]
        single_interpo[-1, :] = z2[i]
        print(single_interpo.shape)
        for interpo in range(interpo_num):
            c = torch.lerp(z1[i], z2[i], interpo/interpo_num)
            single_interpo[interpo,:] = c
        zspace.append(single_interpo)
    return glowmodel.reverse(zspace).cpu().data

if __name__ == "__main__":
    pass