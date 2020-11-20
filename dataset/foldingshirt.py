from PIL import Image
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os

class shirtDataset_train(Dataset):
    def __init__(self, path, transform=None):

        # read the h5 file
        self.full_data = None
        with open(path, 'rb') as f:
            self.full_data = pickle.load(f)

        self.numpair = len(self.full_data)
        
        self.numpair = 1000

        # imagedata
        self.full_imagedata = []
        for i in range(self.numpair):
            self.full_imagedata.append(Image.fromarray(self.full_data[i][0]))
            # self.full_imagedata.append(Image.fromarray(self.full_data[i][1]))

        self.transform = transform

    def __len__(self):
        # the length of valid frames
        return self.numpair

    def __getitem__(self, idx):
        output = self.full_imagedata[idx]

        if self.transform:
            output = self.transform(output)
        return output
    
class shirtDataset_test(Dataset):
    def __init__(self, path, transform=None):

        # read the h5 file
        self.full_data = None
        with open(path, 'rb') as f:
            self.full_data = pickle.load(f)

        self.numpair = len(self.full_data) - 1000

        # imagedata
        self.full_imagedata = []
        for i in range(self.numpair):
            self.full_imagedata.append(Image.fromarray(self.full_data[i+1000][0]))
            # self.full_imagedata.append(Image.fromarray(self.full_data[i+1000][1]))

        self.transform = transform

    def __len__(self):
        # the length of valid frames
        return self.numpair

    def __getitem__(self, idx):
        output = self.full_imagedata[idx]

        if self.transform:
            output = self.transform(output)
        return output

if __name__ == '__main__':
    path = 'data/shirt_dataset_20191217_20200109_no_unf.pkl'
    image_size = 64
    batchsize = 32

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ]
    )

    dataset = shirtDataset(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batchsize, num_workers=4)
    loader = iter(loader)
    data = next(loader)
    print(data.shape)
