#!/bin/bash -l

#SBATCH -o ./out/job.%j
#SBATCH -e ./err/job.%j
#SBATCH -D ./
#SBATCH -J gn
##SBATCH -J fc_mnist
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=1
#SBATCH --time=09:59:59

module purge
module load gcc/10 impi/2019.8
module load anaconda/3/2020.02
module load pytorch/cpu/1.5.0
module load mpi4py/3.0.3

pip install --user tensorflow_addons
pip install --user tensorflow_datasets
pip install --user keras_tuner

# Limit the number of OMP threads to the available resources per MPI task (here: 1 core)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python3 ./jacobian_rank.py

## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
## text to create a longer file
