import os
import numpy as np
import h5py

def aggregate_results(output_dir, world_size):
    aggregated_hdf5_path = os.path.join(output_dir, 'aggregated_results_model_separate_ff_separate_conv_resume2_160_80_10_epoch=10.h5')

    # creat h5 file to store  results
    with h5py.File(aggregated_hdf5_path, 'w') as agg_h5f:
        r_predictions_dset = agg_h5f.create_dataset('r_predictions', (0, 115), maxshape=(None, 115), chunks=True)
        c_predictions_dset = agg_h5f.create_dataset('c_predictions', (0, 115), maxshape=(None, 115), chunks=True)
        r_targets_dset = agg_h5f.create_dataset('r_targets', (0, 115), maxshape=(None, 115), chunks=True)
        c_targets_dset = agg_h5f.create_dataset('c_targets', (0, 115), maxshape=(None, 115), chunks=True)
        params_dset = agg_h5f.create_dataset('params', (0, 4), maxshape=(None, 4), chunks=True)

        def append_to_dataset(dset, data):
            """Helper function to append data to an existing HDF5 dataset."""
            dset.resize(dset.shape[0] + data.shape[0], axis=0)
            dset[-data.shape[0]:] = data

        # loop over each output file and concatenate to aggregated results file
        for i in range(world_size):
            hdf5_path = os.path.join(output_dir, f'predictions_gpu_{i}.h5')
            try:
                with h5py.File(hdf5_path, 'r') as gpu_h5f:
                    r_pred = gpu_h5f['r_predictions'][:]
                    c_pred = gpu_h5f['c_predictions'][:]
                    r_targ = gpu_h5f['r_targets'][:]
                    c_targ = gpu_h5f['c_targets'][:]
                    params = gpu_h5f['params'][:]

                    append_to_dataset(r_predictions_dset, r_pred)
                    append_to_dataset(c_predictions_dset, c_pred)
                    append_to_dataset(r_targets_dset, r_targ)
                    append_to_dataset(c_targets_dset, c_targ)
                    append_to_dataset(params_dset, params)

            except FileNotFoundError:
                print(f"Warning: HDF5 file for GPU {i} not found. Skipping aggregation for this GPU.")

    with h5py.File(aggregated_hdf5_path, 'r') as agg_h5f:
        num_data_points = agg_h5f['r_predictions'].shape[0]
        print(f"Total number of data points aggregated: {num_data_points}")

    '''# Clean up original files
    for i in range(world_size):
        try:
            os.remove(os.path.join(output_dir, f'predictions_gpu_{i}.h5'))
        except FileNotFoundError:
            print(f"File for GPU {i} already removed or not found. Skipping cleanup for this GPU.")'''

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate results from multiple GPUs")
    parser.add_argument('--output_dir', help='Directory containing the GPU-specific result files', default='/projects/bbvf/victoria/Transformer_training/inference/')
    parser.add_argument('--world_size', type=int, help='Total number of GPUs used during inference', default=2)
    args = parser.parse_args()

    aggregate_results(args.output_dir, args.world_size)

