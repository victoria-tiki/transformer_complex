import torch
import torch.utils.data as data
import h5py
import numpy as np
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from functools import partial


class h5Generator(data.Dataset):
    'Generates data for PyTorch'
    def __init__(self, file_path, dim=(10130,), n_channels=2, shuffle=True, normalize=False, return_labels=False, batch_size=32):
        'Initialization'
        self.file_path = file_path
        self.f = h5py.File(self.file_path, 'r')
        self.keys = list(self.f.keys())
        self.dset = self.f[self.keys[0]]
        self.lset = self.f[self.keys[1]]
        
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.normalize = normalize
        self.return_labels = return_labels
        self.indices = np.arange(len(self.dset))
        self.batch_size = batch_size
                        
        self.enc_start, self.enc_end = 5000 // 2, (10000 - 100) // 2
        self.pred_start, self.pred_end = (10000 - 100) // 2, 10130 // 2

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dset)

    def __getitem__(self, index):
        'Generate one sample of data'
        # Find the index of interest
        idx = self.indices[index]
        return idx
        
    @staticmethod
    def custom_collate_fn(batch_indices, dataset):
        
        # Sort indices for h5py compatibility
        sorted_indices = np.sort(batch_indices)
        original_order = np.argsort(batch_indices)

        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((dataset.batch_size, *dataset.dim, dataset.n_channels),dtype=np.float32)  
        y = np.empty((dataset.batch_size, 4), dtype=int)
        
        X[:, :, 0] = dataset.dset[sorted_indices, :].real
        X[:, :, 1] = dataset.dset[sorted_indices, :].imag
        X = X[:, ::2, :]
        y = dataset.lset[sorted_indices, :]
        
        if dataset.normalize:
            X=np.log(1+X)
        
        encoder_input = X[:, dataset.enc_start:dataset.enc_end, :]
        
        decoder_input = X[:, dataset.enc_end-1:-1, :]
        decoder_target = X[:, dataset.enc_end:, :]
        
        encoder_input=torch.from_numpy(encoder_input)
        decoder_input=torch.from_numpy(decoder_input)
        decoder_target=torch.from_numpy(decoder_target)
        labels=torch.from_numpy(y)
        
        if dataset.return_labels:
            return [encoder_input, decoder_input], decoder_target, labels
        else:
            return [encoder_input, decoder_input], decoder_target
        
            
class WaveformDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, normalize=True, num_workers=2):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.normalize = normalize
        self.num_workers= num_workers

    def setup(self, stage='fit'):
        if stage == 'fit' or stage is None:
            self.train_dataset = h5Generator(file_path=os.path.join(self.data_dir, 'train.hdf5'), 
                                             normalize=self.normalize, return_labels=False, batch_size=self.batch_size)
            self.val_dataset = h5Generator(file_path=os.path.join(self.data_dir, 'val.hdf5'), 
                                           normalize=self.normalize, return_labels=False, batch_size=self.batch_size)
        if stage == 'test' or stage is None:
            self.test_dataset = h5Generator(file_path=os.path.join(self.data_dir, 'test.hdf5'), 
                                            normalize=self.normalize, return_labels=True,batch_size=self.batch_size)

    def train_dataloader(self):
        collate_fn = partial(h5Generator.custom_collate_fn, dataset=self.train_dataset)
        train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, pin_memory=True, num_workers = self.num_workers, collate_fn=collate_fn, drop_last= True)

    def val_dataloader(self):
        collate_fn = partial(h5Generator.custom_collate_fn, dataset=self.val_dataset)
        val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=val_sampler, pin_memory=True, num_workers = self.num_workers, collate_fn=collate_fn, drop_last= True)

    def test_dataloader(self):
        collate_fn = partial(h5Generator.custom_collate_fn, dataset=self.test_dataset)
        #test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True,pin_memory=True, num_workers = self.num_workers, collate_fn=collate_fn, drop_last= True)
   