#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # Number of tasks per node (one per GPU)
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --account=bcbw-dtai-gh
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=transf_inference
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

module load python/miniforge3_pytorch/2.5.0
module list

echo "Job is starting on `hostname`"

# Start overall timing
start_time=$(date +%s)

WORLD_SIZE=4
OUTPUT_DIR=/projects/bbvf/victoria/Transformer_training/inference/

for i in $(seq 0 $(($WORLD_SIZE - 1))); do
    echo "Starting inference on GPU $i"
    time srun --ntasks=1 --gres=gpu:1 --exclusive python inference.py \
        --batch_size 128 \
        --data_dir /projects/bbvf/victoria/Transformer_data/ \
        --max_batches 1630 \
        --output_dir $OUTPUT_DIR \
        --gpu_index $i \
        --world_size $WORLD_SIZE &
done

wait

echo "Starting aggregation of results"
time python aggregate_results.py --output_dir $OUTPUT_DIR --world_size $WORLD_SIZE

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

echo "Total time for inference and aggregation: $elapsed_time seconds"
