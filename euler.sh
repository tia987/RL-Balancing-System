#!/bin/bash
#SBATCH --job-name=RSL26            # Job name    (default: sbatch)
#SBATCH --output=rsl.out            # Output file (default: slurm-%j.out)
#SBATCH --error=rsl.err             # Error file  (default: slurm-%j.out)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=16                 # Number of tasks
#SBATCH --ntasks-per-node=16        # Number of tasks per node
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=2G            # Memory per CPU
#SBATCH --time=24:00:00             # Wall clock time limit
#SBATCH --mail-type=END,FAIL        # Send an email when job ends

# Load some modules
module load stack/2025-06 gcc/12.2.0 eth_proxy
module list

export UV_NO_CONFIG=1
export OMNI_KIT_ACCEPT_EULA=YES

# Load environment
source env.sh

# General configuration
TASK="Twowheelbalance-Twb-Rider"

# Train
make train_multi TASK="$TASK" ITERATIONS=4000 NUM_ENVS=1024

# Generate arrow videos
for mass in 0 100; do
    make video_arrow TASK="$TASK" RIDER_MASS="$mass"
done

# Generate terrain videos
# for terrain in flat random_rough slope; do
#     make video_terrain TASK="$TASK" RIDER_MASS=50 TERRAIN="$terrain"
# done

# Evaluate
make eval TASK="$TASK"
