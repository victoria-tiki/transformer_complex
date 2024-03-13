#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu      # <- or one of: cpu, gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcdz-delta-cpu
#SBATCH --job-name=transf_inference
#SBATCH --time=12:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
### GPU options ###
##SBATCH --gpus-per-node=1
##SBATCH --gpu-bind=none     # <- or closest

module load python
module load anaconda3_cpu/23.7.4 #anaconda3_gpu/23.9.0   
module list  # job documentation and metadata

echo "job is starting on `hostname`"
srun $(which python) inference.py --batch_size 8 --data_dir /scratch/bbke/victoria/Transformer_data/ --max_batches 50 --output_dir /scratch/bbke/victoria/Transformer_training/inference/