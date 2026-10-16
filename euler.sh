#!/bin/bash
#SBATCH --job-name=RSL26            # Job name    (default: sbatch)
#SBATCH --output=rsl.out            # Output file (default: slurm-%j.out)
#SBATCH --error=rsl.err             # Error file  (default: slurm-%j.out)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=16                 # Number of tasks
#SBATCH --ntasks-per-node=16        # Number of tasks per node
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=2G            # Memory per CPU
#SBATCH --time=24:00:00             # Wall clock time limit
#SBATCH --mail-type=END,FAIL        # Send an email when job ends

# Load some modules
module load stack/2025-06 gcc/12.2.0 python/3.12.0 eth_proxy
module list

export UV_NO_CONFIG=1
export OMNI_KIT_ACCEPT_EULA=YES

# Load environment
source env.sh

# Run the program
make train_multi ITERATIONS=6000
