#!/bin/bash
#SBATCH --job-name=test
#SBATCH --cluster=gpu
#SBATCH --partition=isenocak
#SBATCH --nodes=1                # node count
#SBATCH --gres=gpu:1     
#SBATCH --cpus-per-task=1            # Request only 1 CPU
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=02-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

crc-job-stats

module reset
module load python/3.12.8
source "$HOME/venvs/torch312/bin/activate"

export TF_CPP_MIN_LOG_LEVEL=3

srun -n 1 python -u main.py
#mpirun --oversubscribe --np 1 python -u main.py

