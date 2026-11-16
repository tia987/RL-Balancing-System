"""Shared-control baseline with a rider BODY (task Twowheelbalance-Twb-Rider).

The robot has an extra link "rider" (the person's upper body) on a hip joint that behaves like a spring-damper
(see assets/tw_balance_rider.py). The simulated person leans to drive and uses the handlebar/joystick to steer:

  - commands.base_velocity is what the rider WANTS: vx (forward speed), wz (turning rate)
  - lean angle the rider aims for = lean_per_speed * vx      (default 0.26 rad = 15 deg per 1 m/s), reached smoothly
  - the velocity reward is unchanged (track commands.base_velocity), so it is proportional to the tilt
  - the policy is not told vx. It sees the measured lean angle of the rider and the joystick (wz).

Use (env_cfg_rider.py):
  events:        rider_body = EventTerm(func=mdp.rider_body_step, mode="interval", interval_range_s=(0.0, 0.0),
                                        params={"asset_cfg": SceneEntityCfg("robot", joint_names="rider_joint")})
  observations:  joystick = ObsTerm(func=mdp.joystick_command),  rider_lean = ObsTerm(func=mdp.rider_lean_obs, ...)
"""
from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def rider_body_step(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="rider_joint"),
    command_name: str = "base_velocity",
    lean_per_speed: float = 0.26,                        # rad of lean per m/s the rider wants (15 deg at 1 m/s)
    lean_max: float = 0.35,                              # rad, the rider never aims further than this
    lean_tau: tuple[float, float] = (0.25, 0.6),         # s, how quickly the rider changes the lean (random per rider)
    sway: float = 0.0,                                   # rad, random body movement that is NOT a command (0 = off)
    sway_tau: float = 1.0,                               # s, how slowly that movement drifts
):
    """Interval event (interval_range_s=(0, 0) -> every policy step, all robots): move the rider's hip-joint target
    towards the lean that belongs to the speed the rider wants. The spring-damper of the joint does the rest."""
    asset = env.scene[asset_cfg.name]
    n, dev = env.num_envs, env.device
    if not hasattr(env, "rider_lean_target"):
        env.rider_lean_target = torch.zeros(n, device=dev)
        env._rider_tau = torch.empty(n, device=dev).uniform_(*lean_tau)
        env._rider_sway = torch.zeros(n, device=dev)
    new = env.episode_length_buf == 0                    # robots reset in this step: a new rider, sitting upright
    if new.any():
        k = int(new.sum())
        env.rider_lean_target[new] = 0.0
        env._rider_sway[new] = 0.0
        env._rider_tau[new] = torch.empty(k, device=dev).uniform_(*lean_tau)
    wanted = env.command_manager.get_command(command_name)[:, 0]
    aim = (lean_per_speed * wanted).clamp(-lean_max, lean_max)
    env.rider_lean_target += (aim - env.rider_lean_target) * (env.step_dt / env._rider_tau).clamp(max=1.0)
    target = env.rider_lean_target
    if sway > 0.0:                                       # slow random drift (Ornstein-Uhlenbeck), a disturbance
        a = env.step_dt / sway_tau
        env._rider_sway += -a * env._rider_sway + sway * (2.0 * a) ** 0.5 * torch.randn(n, device=dev)
        target = target + env._rider_sway
    asset.set_joint_position_target_index(target=target.unsqueeze(1), joint_ids=asset_cfg.joint_ids)


def rider_lean_obs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="rider_joint")) -> torch.Tensor:
    """Measured lean angle of the rider's upper body relative to the seat [rad] (e.g. from a backrest / torso sensor).
    Positive = leaning forward. Shape (num_envs, 1)."""
    asset = env.scene[asset_cfg.name]
    return _t(asset.data.joint_pos)[:, asset_cfg.joint_ids].reshape(env.num_envs, -1)[:, :1]


def joystick_command(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Only the steering part (yaw rate) of the velocity command: what the handlebar / joystick gives the policy."""
    return env.command_manager.get_command(command_name)[:, 2:3]

