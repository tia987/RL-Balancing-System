"""PPO settings for the rider task: same as PPORunnerCfg (rsl_rl_ppo_cfg.py), plus observation normalisation.
Put this file next to rsl_rl_ppo_cfg.py:  .../tasks/twb/config/twb/agents/rsl_rl_ppo_rider_cfg.py

Why obs_normalization=True here: the rider task feeds the network inputs of very different size (wheel speeds up to
~18 rad/s and last actions up to 14, but the rider lean is at most 0.35 rad and the balance pitch about 0.07 rad).
Without normalisation the small inputs barely move the first layer, and the lean is the only thing that tells the
policy how fast the rider wants to go. RSL-RL then keeps a running mean / std of every input (saved in the checkpoint).
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class PPORunnerRiderCfg(PPORunnerCfg):
    experiment_name = "twb_rider"   # separate log folder: logs/rsl_rl/twb_rider
    # teacher: the actor also sees rider mass and balance pitch; only the critic sees what the rider wants
    obs_groups = {"actor": ["policy", "privileged"], "critic": ["policy", "privileged", "intent"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=2.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=True,
    )
