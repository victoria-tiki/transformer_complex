import torch
import torch.utils.data as data
import h5py
import numpy as np
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import ConcatDataset

from functools import partial

class h5Generator(data.Dataset):
    'Generates data for PyTorch'
    def __init__(self, file_path, dim=(10130,), n_channels=2, shuffle=True, normalize=False, return_labels=False, batch_size=32):
        'Initialization'
        self.file_path = file_path
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.normalize = normalize
        self.return_labels = return_labels
        self.batch_size = batch_size
        
        with h5py.File(self.file_path, 'r') as f:
            self.keys = list(f.keys())
            self.total_samples = f[self.keys[0]].shape[0]  
            self.labels = f[self.keys[1]][:, -1]  

        self.indices = np.arange(self.total_samples)

        self.enc_start, self.enc_end = 5000 // 2, (10000 - 100) // 2
        self.pred_start, self.pred_end = (10000 - 100) // 2, 10130 // 2

    def __len__(self):
        'Denotes the total number of samples'
        return self.total_samples

    def __getitem__(self, index):
        'Generate one sample of data'
        with h5py.File(self.file_path, 'r') as f:
            dset = f[self.keys[0]]
            lset = f[self.keys[1]]

            idx = self.indices[index]

            real_part = dset[idx, :].real
            imag_part = dset[idx, :].imag

            # apply normalization if specified
            if self.normalize:
                real_part = np.log(1 + real_part)
                imag_part = np.log(1 + imag_part)

            X = np.stack((real_part, imag_part), axis=-1)[::2, :] 

            encoder_input = X[self.enc_start:self.enc_end, :]
            decoder_input = X[self.enc_end-1:-1, :]
            decoder_target = X[self.enc_end:, :]

            # convert to pytorch tensors
            encoder_input = torch.from_numpy(encoder_input)
            decoder_input = torch.from_numpy(decoder_input)
            decoder_target = torch.from_numpy(decoder_target)
            y = torch.from_numpy(lset[idx, :])

        if self.return_labels:
            return [encoder_input, decoder_input], decoder_target, y
        else: 
            return [encoder_input, decoder_input], decoder_target


class WaveformDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, normalize=False, num_workers=2, test_split=0.2, seed=42):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.normalize = normalize
        self.num_workers = num_workers
        self.test_split = test_split
        self.seed = seed  

    def setup(self, stage='fit'):
        # random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.train_dataset = h5Generator(file_path=os.path.join(self.data_dir, 'train_updated_pi_2_no_chunking.hdf5'), 
                                  normalize=self.normalize, return_labels=False, batch_size=self.batch_size)
        self.val_dataset = h5Generator(file_path=os.path.join(self.data_dir, 'val.hdf5'), 
                                 normalize=self.normalize, return_labels=False, batch_size=self.batch_size)
        self.test_dataset = h5Generator(file_path=os.path.join(self.data_dir, 'test.hdf5'), 
                                 normalize=self.normalize, return_labels=True, batch_size=self.batch_size)
                                 
        if stage=='fit':                     
            print(f"Training dataset size: {len(self.train_dataset)} samples")
            print(f"Test dataset size: {len(self.test_dataset)} samples")
            print(f"Validation dataset size: {len(self.val_dataset)} samples \n")


    def train_dataloader(self):
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        #self.train_sampler  = DistributedBiasedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers = self.num_workers, pin_memory=True, prefetch_factor=2)

    def val_dataloader(self):
        val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=val_sampler, num_workers = self.num_workers, pin_memory=True, prefetch_factor=2)

    def test_dataloader(self):
        #test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers, pin_memory=True, prefetch_factor=2)
        
