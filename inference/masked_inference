

import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import argparse
import sys
from torch.utils.data import Subset
#import matplotlib.pyplot as plt

#torch.set_float32_matmul_precision('high')


sys.path.append("/home/victoria/Transformer_training")
from model_separate import create_transformer
from data_generators import h5Generator

def mask_sequence(sequence, mask_type='zeros', mask_range=None, device='cpu'):
    sequence = sequence.clone().to(device)
    if mask_range is None or mask_range[0] == mask_range[1]:
        return sequence  # no masking
    if mask_type == 'zeros':
        sequence[:, mask_range[0]:mask_range[1], :] = 0.0
    elif mask_type == 'random':
        mean, std = sequence.mean(), sequence.std()
        shape = sequence[:, mask_range[0]:mask_range[1], :].shape
        sequence[:, mask_range[0]:mask_range[1], :] = torch.normal(mean, std, size=shape).to(device)
    return sequence
    
def predict_sequences(model, encoder_input, max_len=90):
    encoder_input = encoder_input.clone()#.to(map_location)
    d = encoder_input[:, -1:]#.to(map_location)
    
    for _ in range(max_len):
        with torch.no_grad():
            p = model(encoder_input, d)
            p = p[:, -1:]
        d = torch.cat([d, p], dim=1)
    return d[:, 1:]

def predict_sequences_with_precomputed_encoder(model, encoder_input, max_len=90):
    device = model.conv1d.weight.device
    encoder_input = encoder_input.to(device)
    batch_size = encoder_input.shape[0]

    # Extract real/imag and compute imag_mask
    encoder_real = encoder_input[:, :, 0].unsqueeze(-1)
    encoder_imag = encoder_input[:, :, 1].unsqueeze(-1)
    imag_mask = (encoder_imag <= 1e-6).all(dim=1).float().unsqueeze(-1)

    # Embed + encode
    with torch.inference_mode():
        encoder_embedded_real = model.embedding1(encoder_real)
        encoder_embedded_imag = model.embedding1(encoder_imag)
        encoder_inputs_cat = torch.cat((encoder_embedded_real, encoder_embedded_imag), dim=-1)
        encoder_outputs = model.encoder(encoder_inputs_cat, additional_mask=imag_mask)

    d = encoder_input[:, -1:].clone()

    for i in range(1, max_len + 1):
        with torch.inference_mode():
            p = model(encoder_outputs, d, encoder_is_precomputed=True, imag_mask=imag_mask)
            d = torch.cat([d, p[:, -1:]], dim=1)

    return d[:, 1:]


def run_masked_inference(gpu_index, mask_range, args, mask_ranges):
    torch.cuda.set_device(gpu_index)
    device = torch.device(f'cuda:{gpu_index}')
    print(f"[GPU {gpu_index}] Running inference for mask range {mask_range}")

    test_dataset = h5Generator(
        file_path=os.path.join(args.data_dir, 'test.hdf5'),
        normalize=True,
        return_labels=True,
        batch_size=args.batch_size
    )
    

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = create_transformer(embed_dim=160 // 2, dense_dim=80, num_heads=10, device=device)
    model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()})
    model.to(device)
    #model = torch.compile(model, mode="reduce-overhead")
    model.eval()

    predictions_r, predictions_c = [], []
    targets_r, targets_c, label_params = [], [], []
    

    pbar = tqdm(total=len(test_loader), desc=f"[GPU {gpu_index}] Mask {mask_range}", position=gpu_index)

    with torch.no_grad():
        for i, ([encoder_input, decoder_input], decoder_target, labels) in enumerate(test_loader):
            encoder_input = encoder_input.to(device)
            decoder_target = decoder_target.to(device)
            labels = labels.to(device)

            masked_input = mask_sequence(encoder_input, mask_type=args.mask_type, mask_range=mask_range, device=device)
            pred = predict_sequences_with_precomputed_encoder(model, masked_input, max_len=115)

            predictions_r.append(pred[:, :, 0].cpu().numpy())
            predictions_c.append(pred[:, :, 1].cpu().numpy())

            if mask_range == (0, 0):
                targets_r.append(decoder_target[:, :, 0].cpu().numpy())
                targets_c.append(decoder_target[:, :, 1].cpu().numpy())
                label_params.append(labels.cpu().numpy())

            if args.max_batches and i >= args.max_batches - 1:
                break
            pbar.update(1)

    pbar.close()

    os.makedirs(args.output_dir, exist_ok=True)
    key = f"{mask_range[0]}_{mask_range[1]}"
    np.save(os.path.join(args.output_dir, f"r_predictions_masked_{key}.npy"), np.vstack(predictions_r))
    np.save(os.path.join(args.output_dir, f"c_predictions_masked_{key}.npy"), np.vstack(predictions_c))

    if mask_range == (0, 0):
        np.save(os.path.join(args.output_dir, f"r_targets.npy"), np.vstack(targets_r))
        np.save(os.path.join(args.output_dir, f"c_targets.npy"), np.vstack(targets_c))
        np.save(os.path.join(args.output_dir, f"params.npy"), np.vstack(label_params))

    print(f"[GPU {gpu_index}] Done with mask range {mask_range}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='/home/victoria/Transformer_training/inference/ckpt/model.ckpt')
    parser.add_argument('--data_dir', default='/home/victoria/Transformer_data')
    parser.add_argument('--output_dir', default='/home/victoria/Transformer_training/inference/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--mask_type', choices=['zeros', 'random'], default='zeros')
    parser.add_argument('--mask_index', type=int, required=True, help="Which mask range to use [0-5]")

    args = parser.parse_args()

    mask_ranges = [
        (0, 0),        
        (0, 500),
        (500, 1000),
        (1000, 1500),
        (1500, 2000),
        (2000, 2400),
        (2400, 2440)
    ]
    

    if args.mask_index < 0 or args.mask_index >= len(mask_ranges):
        raise ValueError(f"Invalid mask_index {args.mask_index}. Must be 0–{len(mask_ranges) - 1}.")

    run_masked_inference(0, mask_ranges[args.mask_index], args, mask_ranges)

if __name__ == '__main__':
    main()
