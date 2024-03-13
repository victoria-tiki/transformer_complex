#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.append("/scratch/bbke/victoria/Transformer_training/")

from models import create_transformer  
from data_generators import *  
from time import time
from tqdm import tqdm
import os
from os import path, makedirs

def inference(checkpoint_path, data_dir, output_dir, batch_size=16,max_batches=None, device='cpu'):
    
    dataset = WaveformDataModule(data_dir, batch_size=batch_size, normalize=True)
    dataset.setup(stage='test')
    test_dataset=dataset.test_dataloader()

    # Loading the trained model from  checkpoint
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model = create_transformer(embed_dim=128, dense_dim=64, num_heads=8,device=device)
    
    # Modify state_dict keys before loading
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('model.', '')  # Remove 'model.' prefix
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model=model.to(device)
    model.eval()
    
    t0 = time()
    
    predictions, targets, params = [], [], []
    
    def predict_sequences(model, encoder_input, max_len=90, verbose=False):
        encoder_input = encoder_input.clone()
        d = encoder_input[:, -1:]
        
        for i in range(max_len):
            with torch.no_grad():
                p = model(encoder_input, d)
                p=p[:, -1:]
            d = torch.cat([d, p], dim=1)

        return d[:, 1:]


    c_predictions = []
    c_targets = []
    r_predictions = []
    r_targets = []
    params = []
    
    i = 0
    with tqdm(total=max_batches, desc="Inference", unit="batch") as pbar:
        for [encoder_input, decoder_input], decoder_target, labels in test_dataset:
            
    
            pred = predict_sequences(model, encoder_input, max_len=115)
            
            pred_r, pred_c= pred[:,:,0],pred[:,:,1]
            decoder_target_r,decoder_target_c=decoder_target[:,:,0],decoder_target[:,:,1]
    
            r_predictions.append(pred_r.detach().numpy())
            c_predictions.append(pred_c.detach().numpy())
            r_targets.append(decoder_target_r.numpy())
            c_targets.append(decoder_target_c.numpy())
            params.append(labels.numpy())
            
            torch.cuda.empty_cache()
            
            pbar.update(1)
            sys.stdout.flush()
            
            i += 1
            if i > max_batches-1:
                break
    
    r_predictions = np.vstack(r_predictions)
    c_predictions = np.vstack(c_predictions)
    r_targets = np.vstack(r_targets)
    c_targets = np.vstack(c_targets)
    params = np.vstack(params)
    
    print('Number of predicted waveforms:',r_predictions.shape[0])
    
    np.save(output_dir + 'c_predictions.npy', c_predictions)
    np.save(output_dir + 'c_targets.npy', c_targets)
    np.save(output_dir + 'r_predictions.npy', r_predictions)
    np.save(output_dir + 'r_targets.npy', r_targets)
    np.save(output_dir + 'params.npy', params)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waveform data inference")
    parser.add_argument('--checkpoint_path', help='path to the saved model checkpoint', default='/scratch/bbke/victoria/Transformer_training/checkpoint/weights/model_epoch=00-val_loss=0.00001.ckpt')
    parser.add_argument('--data_dir', help='path to the test data', default='/scratch/bbke/victoria/Transformer_data')
    parser.add_argument('--output_dir', help='directory to save predictions and targets', default='/scratch/bbke/victoria/Transformer_training/inference/')
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--max_batches', type=int, default=20)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference(args.checkpoint_path, args.data_dir, args.output_dir, args.batch_size, args.max_batches, device)
