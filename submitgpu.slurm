#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4  #should be equal to gpus-per-node
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: cpu, gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcdz-delta-gpu
#SBATCH --job-name=transformer
#SBATCH --time=00:30:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=4
##SBATCH --gpu-bind=none     # <- or closest

module load python
module load anaconda3_gpu/23.9.0    #anaconda3_cpu/23.7.4
module list  # job documentation and metadata

export OMP_NUM_THREADS=16

echo "job is starting on `hostname`"
srun -n $((SLURM_NNODES * SLURM_NTASKS_PER_NODE)) -N $SLURM_NNODES -c $SLURM_CPUS_PER_TASK $(which python) train.py --batch_size 64 --data_dir /scratch/bbke/victoria/Transformer_data/ --checkpoint_dir /scratch/bbke/victoria/Transformer_training/checkpoint/ --num_nodes $SLURM_JOB_NUM_NODES --num_workers 8