"""PPO settings for the rider-body task: same as PPORunnerCfg, other observation groups and log folder.
Put this file next to rsl_rl_ppo_cfg.py:  .../tasks/twb/config/twb/agents/rsl_rl_ppo_rider_cfg.py
"""
from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class PPORunnerRiderCfg(PPORunnerCfg):
    experiment_name = "twb_rider"   # separate log folder: logs/rsl_rl/twb_rider
    # teacher: the actor also sees rider mass and balance pitch; only the critic sees what the rider wants
    obs_groups = {"actor": ["policy", "privileged"], "critic": ["policy", "privileged", "intent"]}

