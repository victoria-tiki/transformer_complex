#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.append("/projects/bbvf/victoria/Transformer_training")
from models_weights import create_transformer  
from data_generators import * 

def apply_log_transform(attn_weights, epsilon=1e-10):
    """
    Apply logarithmic transformation to the attention weights.
    """
    attn_weights_clipped = np.clip(attn_weights, a_min=epsilon, a_max=None)  
    attn_weights_log = np.log(attn_weights_clipped)
    return attn_weights_log


def plot_self_attention_heatmap_with_waveforms(attn_weights, encoder_real_waveform, encoder_imag_waveform, decoder_real_waveform, decoder_imag_waveform, title, output_dir):
    font_size = 18
    """Plot self-attention heatmap with corresponding waveforms."""
    decoder_timesteps = np.arange(-100, 130, 2)  
    encoder_timesteps = decoder_timesteps  
    manual_ticks_y = [-100, -50, 0, 50, 100]  

    fig = plt.figure(figsize=(7.5, 10)) 
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 5], height_ratios=[1, 5, 0.5], hspace=0.1, wspace=0.1)  

    # upper left (empty)
    ax_empty = fig.add_subplot(gs[0, 0])
    ax_empty.axis('off')  

    # upper right (decoder waveform)
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.plot(decoder_timesteps, decoder_real_waveform.squeeze().cpu().numpy(), color='blue', label=r'$h_+$')
    ax_top.plot(decoder_timesteps, decoder_imag_waveform.squeeze().cpu().numpy(), color='red', linestyle='dashed', label=r'$h_{\times}$')
    ax_top.set_xlim([decoder_timesteps[0], decoder_timesteps[-1]])
    ax_top.axis('off')  
    ax_top.legend(loc='upper right', bbox_to_anchor=(1, 1.3), fontsize=font_size)  

    # lower left (encoder waveform, rotated -90 degrees)
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.plot(encoder_real_waveform.squeeze().cpu().numpy(), encoder_timesteps, color='blue', label=r'$h_+$')
    ax_left.plot(encoder_imag_waveform.squeeze().cpu().numpy(), encoder_timesteps, color='red', linestyle='dashed', label=r'$h_{\times}$')
    ax_left.set_ylim([encoder_timesteps[0], encoder_timesteps[-1]])
    ax_left.set_ylabel('Encoder Timesteps', rotation=-90, labelpad=30, fontsize=font_size)  
    ax_left.invert_yaxis()  
    ax_left.invert_xaxis()
    ax_left.axis('off')  
    ax_left.legend(loc='upper left', bbox_to_anchor=(-0.65, 0.35), labelspacing=0.3, fontsize=font_size)  
    ax_left.set_position([0.12, 0.29, 0.1, 0.45])

    # main heatmap
    ax_heatmap = fig.add_subplot(gs[1, 1])
    sns.heatmap(
        apply_log_transform(np.transpose(attn_weights.cpu().detach().numpy())),  
        cmap='viridis',
        xticklabels=decoder_timesteps,
        yticklabels=encoder_timesteps,
        ax=ax_heatmap,
        cbar=False  
    )
    
    ax_heatmap.set_position([0.27, 0.29, 0.63, 0.45])  
    ax_heatmap.yaxis.tick_right()
    ax_heatmap.yaxis.set_label_position("right")
    
    manual_ticks_x = [-100, -50, 0, 50, 100]  
    x_tick_positions = [np.argmin(np.abs(decoder_timesteps - tick)) for tick in manual_ticks_x]
    ax_heatmap.set_xticks(x_tick_positions)
    ax_heatmap.set_xticklabels([str(decoder_timesteps[i]) for i in x_tick_positions], rotation=0, fontsize=font_size)

    y_tick_positions = [np.argmin(np.abs(encoder_timesteps - tick)) for tick in manual_ticks_y]
    ax_heatmap.set_yticks(y_tick_positions)
    ax_heatmap.set_yticklabels([str(encoder_timesteps[i]) for i in y_tick_positions], fontsize=font_size)
    ax_heatmap.set_xlabel('Decoder Timesteps', labelpad=10, fontsize=font_size)  
    ax_heatmap.set_ylabel('Decoder Timesteps', labelpad=10, fontsize=font_size
    
    cbar_ax = fig.add_axes([0.27, 0.19, 0.63, 0.02])  
    cbar = fig.colorbar(ax_heatmap.collections[0], cax=cbar_ax, orientation='horizontal')
    cbar_ax.set_xlabel('Attention Weights', fontsize=font_size)
    log_ticks = np.linspace(apply_log_transform(attn_weights.cpu().detach().numpy()).min(), apply_log_transform(attn_weights.cpu().detach().numpy()).max(), num=5)
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f'$10^{{{int(tick)}}}$' for tick in log_ticks])
    cbar.ax.tick_params(labelsize=font_size)  
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, title + '.png'))
    plt.close()

def plot_cross_attention_heatmap_with_waveforms(attn_weights, encoder_real_waveform, encoder_imag_waveform, decoder_real_waveform, decoder_imag_waveform, title, output_dir):
    font_size = 18
    """Plot cross-attention heatmap with corresponding waveforms."""
    
    decoder_timesteps = np.arange(-100, 130, 2) 
    encoder_timesteps = np.arange(-5000, -100, 2)  
    
    fig = plt.figure(figsize=(7.5, 10)) 
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 5], height_ratios=[1, 5, 0.5], hspace=0.1, wspace=0.1)

    # upper right (decoder waveform)
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.plot(decoder_timesteps, decoder_real_waveform.squeeze().cpu().numpy(), color='blue', label=r'$h_+$')
    ax_top.plot(decoder_timesteps, decoder_imag_waveform.squeeze().cpu().numpy(), color='red', linestyle='dashed', label=r'$h_{\times}$')
    ax_top.set_xlim([decoder_timesteps[0], decoder_timesteps[-1]])
    ax_top.axis('off')  
    ax_top.legend(loc='upper right', bbox_to_anchor=(1, 1.3), fontsize=font_size)  
    
    # lower left (encoder waveform, rotated -90 degrees)
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.plot(encoder_real_waveform.squeeze().cpu().numpy(), encoder_timesteps, color='blue', label=r'$h_+$')
    ax_left.plot(encoder_imag_waveform.squeeze().cpu().numpy(), encoder_timesteps, color='red', linestyle='dashed', label=r'$h_{\times}$')
    ax_left.set_ylim([encoder_timesteps[0], encoder_timesteps[-1]])
    ax_left.set_ylabel('Encoder Timesteps', rotation=-90, labelpad=30, fontsize=font_size)  
    ax_left.invert_yaxis()  
    ax_left.invert_xaxis()
    ax_left.axis('off')  
    ax_left.legend(loc='upper left', bbox_to_anchor=(-0.65, 0.35), fontsize=font_size) 
    ax_left.set_position([0.12, 0.29, 0.1, 0.45])

    attn_weights_df = pd.DataFrame(
        apply_log_transform(np.transpose(attn_weights.cpu().detach().numpy())),  
        index=encoder_timesteps, 
        columns=decoder_timesteps 
    )
    
    # Main heatmap
    ax_heatmap = fig.add_subplot(gs[1, 1])
    sns.heatmap(
        attn_weights_df,  
        cmap='viridis',
        xticklabels=25,  
        yticklabels=500,  
        ax=ax_heatmap,
        cbar=False
    )
    
    ax_heatmap.set_position([0.27, 0.29, 0.63, 0.45])  
    ax_heatmap.yaxis.tick_right()
    ax_heatmap.yaxis.set_label_position("right")
    ax_heatmap.set_xlabel('Decoder Timesteps', labelpad=10, fontsize=font_size)  
    ax_heatmap.set_ylabel('Encoder Timesteps', labelpad=10, fontsize=font_size)  
    ax_heatmap.tick_params(axis='x', labelsize=font_size)
    ax_heatmap.tick_params(axis='y', labelsize=font_size)
    cbar_ax = fig.add_axes([0.27, 0.19, 0.63, 0.02])  
    cbar = fig.colorbar(ax_heatmap.collections[0], cax=cbar_ax, orientation='horizontal')
    cbar_ax.set_xlabel('Attention Weights', fontsize=font_size) 
    
    log_ticks = np.linspace(apply_log_transform(attn_weights.cpu().detach().numpy()).min(), apply_log_transform(attn_weights.cpu().detach().numpy()).max(), num=5)
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f'$10^{{{int(tick)}}}$' for tick in log_ticks])
    cbar.ax.tick_params(labelsize=font_size)  
    
    plt.savefig(os.path.join(output_dir, title + '.png'))
    plt.close()


def inference(checkpoint_path, data_dir, output_dir, batch_size=16,max_batches=None, device='cpu',plot_weights=False):
    
    start_time = time()
        
    dataset = WaveformDataModule(data_dir, batch_size=batch_size, normalize=True, num_workers=16)
    dataset.setup(stage='fit')
    test_dataset=dataset.train_dataloader()

    # load trained model
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model = create_transformer(embed_dim = 160//2,dense_dim = 80,num_heads = 10,device=device)
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('model.', '')  # Remove 'model.' prefix
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model=model.to(device)
    model.eval()
    
    base_output_dir = os.path.join(output_dir, 'batch_plots')
    os.makedirs(base_output_dir, exist_ok=True)

    for i, ([encoder_input, decoder_input], decoder_target, labels) in enumerate(test_dataset):
        encoder_inputs = encoder_input.to(device)
        decoder_inputs = decoder_input.to(device)
        decoder_target = decoder_target.to(device)
        labels = labels.to(device)

        # Apply filtering based on certain criteria
        condition_a = (labels[:, 0] > 7.6) & (labels[:, 1] > 0.78) & (labels[:, 2] > 0.78)
        condition_b = (labels[:, 0] > 7.6) & (labels[:, 1] > 0.78) & (labels[:, 2] < -0.78)
        selected_indices = torch.where(condition_a | condition_b)[0]  # Get indices of samples to plot
        
        # Run a forward pass to get outputs and attention weights
        with torch.no_grad():
            decoder_outputs, encoder_attn_weights, decoder_self_attn_weights, cross_attn_weights = model(
                encoder_inputs, decoder_inputs, return_weights=True
            )
        
        # Plot and save attention heatmaps with waveforms for each selected element in the batch
        for j in selected_indices:
            label_dir_name = f'{labels[j,0]:.2f}_{labels[j,1]:.2f}_{labels[j,2]:.2f}'
            individual_output_dir = os.path.join(base_output_dir, label_dir_name)
            os.makedirs(individual_output_dir, exist_ok=True) 
            
            for k in range(0, 10):
                plot_title_cross = f'crossattention_head{k}_{labels[j,0]:.2f}_{labels[j,1]:.2f}_{labels[j,2]:.2f}'
                plot_title_self = f'maskedselfattention_head{k}_{labels[j,0]:.2f}_{labels[j,1]:.2f}_{labels[j,2]:.2f}'
                plot_title_enc = f'encoderselfattention_head{k}_{labels[j,0]:.2f}_{labels[j,1]:.2f}_{labels[j,2]:.2f}'

                # Cross-Attention Plot
                plot_cross_attention_heatmap_with_waveforms(
                    cross_attn_weights[j, k],
                    encoder_inputs[j, :, 0], encoder_inputs[j, :, 1],
                    decoder_inputs[j, :, 0], decoder_inputs[j, :, 1],
                    plot_title_cross,
                    individual_output_dir
                )

                # Self-Attention Plot
                plot_self_attention_heatmap_with_waveforms(
                    decoder_self_attn_weights[j, k],
                    decoder_inputs[j, :, 0], decoder_inputs[j, :, 1],
                    decoder_inputs[j, :, 0], decoder_inputs[j, :, 1],
                    plot_title_self,
                    individual_output_dir
                )                

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waveform data inference")
    parser.add_argument('--checkpoint_path', help='path to the saved model checkpoint', default='/projects/bbvf/victoria/Transformer_training/inference/model_separate_ff_separate_conv_resume2_160_80_10_epoch=10-val_loss=0.00000.ckpt')
    parser.add_argument('--data_dir', help='path to the test data', default='/projects/bbvf/victoria/Transformer_data')
    parser.add_argument('--output_dir', help='directory to save predictions and targets', default='/projects/bbvf/victoria/Transformer_training/inference/')
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--max_batches', type=int, default=8)
    parser.add_argument('--plot_weights',type=bool, default=False)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('inference on model located at ',args.checkpoint_path)
    
    print('batch size:',args.batch_size)
    print('number of batches:',args.max_batches)

    inference(args.checkpoint_path, args.data_dir, args.output_dir, args.batch_size, args.max_batches, device, args.plot_weights)