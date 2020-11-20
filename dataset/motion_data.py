import numpy as np
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, data):
        self.n_channels = 3
        self.n_samples = len(data)
        self.data = data.reshape(self.n_samples, self.n_channels, -1)

        self.joints_data = self.data[:,:,:-1]
        self.cont_data = self.data[:,:,-1]
        pass
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        sample = {
            'joints': self.joints_data[index,:,:],
            'control': self.cont_data[index,:]
        }
        return sample

class TestDataset(Dataset):
    def __init__(self, data):
        self.n_channels = 3
        self.n_samples = len(data)
        self.length = len(data[0])
        self.data = data.reshape(self.n_samples, self.length, self.n_channels, -1)

        self.joints_data = self.data[:,:,:,:-1]
        self.cont_data = self.data[:,:,:,-1]
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        sample = {
            'joints': self.joints_data[index,:,:,:],
            'control': self.cont_data[index,:,:]
        }
        return sample