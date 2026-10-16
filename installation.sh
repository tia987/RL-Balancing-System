#!/bin/bash

# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv through python3.12 for isaac lab compatibility
uv venv --python 3.12 .env_isaaclab
source .env_isaaclab/bin/activate

# Install Isaac Sim
uv pip install "isaacsim[all,extscache]==6.1.0" --extra-index-url https://pypi.nvidia.com

# Install isaac Lab
git clone git@github.com:leggedrobotics/IsaacLab-Internal.git IsaacLab
cd IsaacLab
./isaaclab.sh --install
# ./isaaclab.sh --new # Setup new external project, follow instructions

# cp IsaacLab/scripts/reinforcement_learning/play.py ../TwoWheelBalance/scripts
# cp IsaacLab/scripts/reinforcement_learning/train.py ../TwoWheelBalance/scripts

# # Save anymal task in tasks to check out for baseline
# cd ~/RL-Balancing-System
# SRC=IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity
# DST=TwoWheelBalance/src/TwoWheelBalance/tasks
#
# cp -r $SRC/config/anymal_d $DST/anymal_d
# cp $SRC/velocity_env_cfg.py $DST/velocity_env_cfg.py

# @Note: after creating the new template, in the new folder there will be a git folder
# which should be deleted

# Install working folder
uv pip install --no-deps -e TwoWheelBalance

# moviepy 2.0 doesn't work in this scenario
uv pip install "moviepy<2.0"

uv pip install tensorboard tensorflow
