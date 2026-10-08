#!/bin/bash

# Launch environment
source .env_isaaclab/bin/activate
python3 IsaacLab/scripts/tutorials/00_sim/create_empty.py --viz kit # none for headless

cd TwoWheelBalance
python scripts/reinforcement_learning/play.py --task IsaacContrib-Velocity-Rough-AnymalC --rl_library rsl_rl --viz kit
