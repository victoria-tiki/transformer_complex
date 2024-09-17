#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm
import os
from os import path, makedirs
import gc
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


import sys

sys.path.append("/projects/bbvf/victoria/Transformer_training")
from models import create_transformer  
from data_generators import * 

    

def inference(checkpoint_path, data_dir, output_dir, batch_size=16, max_batches=None, device='cpu', rank=0, world_size=1, plot_weights=False):
    start_time = time()
    
    map_location = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Define dataset
    dataset = WaveformDataModule(data_dir, batch_size=batch_size, normalize=True, num_workers=16)
    dataset.setup(stage='test')
    test_dataset = dataset.test_dataloader().dataset

    # splot up across gpus
    num_samples = len(test_dataset)
    samples_per_gpu = num_samples // world_size
    start_index = rank * samples_per_gpu
    end_index = start_index + samples_per_gpu if rank < world_size - 1 else num_samples
    indices = list(range(start_index, end_index))
    test_subset = Subset(test_dataset, indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = create_transformer(embed_dim=160//2, dense_dim=80, num_heads=10, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    
    # Modify state_dict keys before loading (remove superfluous model. prefix)
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('model.', '')  
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model = model.to(map_location)
    model.eval()

    individual_output_dir = os.path.join(output_dir, f'batch_plots')
    os.makedirs(individual_output_dir, exist_ok=True)
    
    #define autoregressive prediction
    def predict_sequences(model, encoder_input, max_len=90):
        encoder_input = encoder_input.clone().to(map_location)
        d = encoder_input[:, -1:].to(map_location)
        
        for _ in range(max_len):
            with torch.no_grad():
                p = model(encoder_input, d)
                p = p[:, -1:]
            d = torch.cat([d, p], dim=1)
        return d[:, 1:]
    
    hdf5_path = os.path.join(output_dir, f'predictions_gpu_{rank}.h5')
    with h5py.File(hdf5_path, 'w') as h5f:
        r_predictions_dset = h5f.create_dataset('r_predictions', (0, 115), maxshape=(None, 115), chunks=True)
        c_predictions_dset = h5f.create_dataset('c_predictions', (0, 115), maxshape=(None, 115), chunks=True)
        r_targets_dset = h5f.create_dataset('r_targets', (0, 115), maxshape=(None, 115), chunks=True)
        c_targets_dset = h5f.create_dataset('c_targets', (0, 115), maxshape=(None, 115), chunks=True)
        params_dset = h5f.create_dataset('params', (0, 4), maxshape=(None, 4), chunks=True)

        def append_to_dataset(dset, data):
            """Helper function to append data to an existing HDF5 dataset."""
            dset.resize(dset.shape[0] + data.shape[0], axis=0)
            dset[-data.shape[0]:] = data

        # inference
        with torch.no_grad():
            pbar = tqdm(total=max_batches, desc="Inference", unit="batch") if rank == 0 else None

            for i, ([encoder_input, decoder_input], decoder_target, labels) in enumerate(test_loader):
                encoder_input = encoder_input.to(map_location)
                decoder_input = decoder_input.to(map_location)
                decoder_target = decoder_target.to(map_location)
                labels = labels.to(map_location)

                # predict entire sequence
                pred = predict_sequences(model, encoder_input, max_len=115)

                pred_r, pred_c = pred[:, :, 0].cpu().numpy(), pred[:, :, 1].cpu().numpy()
                decoder_target_r, decoder_target_c = decoder_target[:, :, 0].cpu().numpy(), decoder_target[:, :, 1].cpu().numpy()
                labels_np = labels.cpu().numpy()

                # Append data to dataset
                append_to_dataset(r_predictions_dset, pred_r)
                append_to_dataset(c_predictions_dset, pred_c)
                append_to_dataset(r_targets_dset, decoder_target_r)
                append_to_dataset(c_targets_dset, decoder_target_c)
                append_to_dataset(params_dset, labels_np)

                del encoder_input, decoder_input, decoder_target, labels, pred
                torch.cuda.empty_cache()
                gc.collect()

                if rank == 0:
                    pbar.update(1)
                    sys.stdout.flush()

                if max_batches and i >= max_batches:
                    break

            if rank == 0:
                pbar.close()

    if rank == 0:
        end_time = time()
        print(f"\nInference completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waveform data inference")
    parser.add_argument('--checkpoint_path', help='Path to the saved model checkpoint', default='/projects/bbvf/victoria/Transformer_training/checkpoint/weights/pi2_separation/separate_ff_and_convolution/model_separate_ff_separate_conv_resume2_160_80_10_epoch=10-val_loss=0.00000.ckpt')
    parser.add_argument('--data_dir', help='Path to the test data', default='/projects/bbvf/victoria/Transformer_data')
    parser.add_argument('--output_dir', help='Directory to save predictions and targets', default='/projects/bbvf/victoria/Transformer_training/inference/')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
    parser.add_argument('--max_batches', type=int, default=8)
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of the GPU being used')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of GPUs being used')
    parser.add_argument('--plot_weights', type=bool, default=False, help='whether to plot attention weights')
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(args.checkpoint_path)
    if args.gpu_index == 0:
        print('batch size:', args.batch_size)
        print('number of batches:', args.max_batches)

    inference(args.checkpoint_path, args.data_dir, args.output_dir, args.batch_size, args.max_batches, device, args.gpu_index, args.world_size, args.plot_weights)
