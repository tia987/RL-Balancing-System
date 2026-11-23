# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Twowheelbalance-Twb-Twb",
    entry_point="TwoWheelBalance.tasks.twb.twb_env:TwbEnv",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv", # Old setup without visualizations
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TwbEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DistillationRunnerCfg",
    },
)

# Shared-control baseline: rider body on a spring-damper hip joint, leans to drive (see env_cfg_rider.py)
gym.register(
    id="Twowheelbalance-Twb-Rider",
    entry_point="TwoWheelBalance.tasks.twb.twb_env:TwbEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_rider:TwbRiderEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_rider_cfg:PPORunnerRiderCfg",
        # TODO: add distillation in the future
    },
)

