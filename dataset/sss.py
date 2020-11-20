from PIL import Image

from torch.utils.data import Dataset, DataLoader
# import torch
from torchvision import transforms
# import os
import glob 
import sys

class sssDataset(Dataset):
    def __init__(self, path, transform=None):

        # read the png file
        self.file_names = glob.glob(path+"/*.png") # list
        

        self.num_images = len(self.file_names)
        

        self.transform = transform

    def __len__(self):
        
        return self.num_images

    def __getitem__(self, idx):
        output = Image.open(self.file_names[idx]).convert("L")
        #print(output.size)
        if self.transform:
            output = self.transform(output)
        return output

if __name__ == '__main__':
    path = sys.argv[1]
    image_size = 128
    batchsize = 32

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (1)),
        ]
    )

    dataset = sssDataset(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batchsize, num_workers=4)
    loader = iter(loader)
    data = next(loader)
    print(data.shape)
