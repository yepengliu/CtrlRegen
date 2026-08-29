#!/bin/bash
# Example Slurm submission. Partition names, GPU types and module commands are
# cluster-specific -- replace the placeholders below with your site's values.
#SBATCH --job-name=ctrlregen-train
#SBATCH --output=ctrlregen-train.log
#SBATCH --nodes=1
#SBATCH --partition=<YOUR_GPU_PARTITION>
#SBATCH --gpus=<GPU_TYPE>:8
#SBATCH --mem=128gb
#SBATCH --time=3-00:00:00

module load cuda   # if your cluster uses environment modules

bash run_semantic.sh    # or: bash run_spatial.sh
