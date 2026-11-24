"""Shared-control baseline with a rider body: task "Twowheelbalance-Twb-Rider" (meeting 9.2, task 3).

Same terrain, rewards, terminations and commands as TwbEnvCfg. What changes:
  - robot: Genny-like chair + 1.75 m mannequin (assets/tw_balance_rider.py): "rider_legs" fixed on the seat,
    "rider" (head, trunk, arms) on a spring-damper hip joint
  - the rider leans to drive: lean angle = 0.26 rad per m/s of the speed the rider wants (mdp.rider_body_step).
    The velocity reward still tracks commands.base_velocity, so it is proportional to the tilt.
  - policy input: usual sensors + joystick (turning rate) + measured lean angle. It is NOT told the forward command.
  - teacher extras (actor and critic): rider mass, balance pitch. Critic only: what the rider wants (velocity command).
Put this file next to env_cfg.py:  TwoWheelBalance/src/TwoWheelBalance/tasks/twb/config/twb/env_cfg_rider.py
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from TwoWheelBalance.assets.tw_balance_rider import BALANCE_BOT_RIDER_CFG
from TwoWheelBalance.tasks.twb import mdp

from .env_cfg import EventCfg, TwbEnvCfg



def _wheels() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])


def _hip() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names="rider_joint")


@configclass
class RiderObservationsCfg:
    """Observation specifications for the rider-body task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """What the chair can measure (16 values)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _wheels()})       # wheels only
        actions = ObsTerm(func=mdp.last_action)
        # handlebar / joystick: left and right only. The forward command is NOT given to the policy.
        joystick = ObsTerm(func=mdp.joystick_command, params={"command_name": "base_velocity"})
        # how far the rider leans (rad, + = forward), with sensor noise of about 1 deg
        rider_lean = ObsTerm(func=mdp.rider_lean_obs, params={"asset_cfg": _hip()}, noise=Unoise(n_min=-0.02, n_max=0.02))

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Teacher information about the rider (actor and critic during training, removed later by distillation)."""

        rider_mass = ObsTerm(func=mdp.rider_mass_obs)
        balance_pitch = ObsTerm(func=mdp.balance_pitch_obs)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class IntentCfg(ObsGroup):
        """What the rider wants (critic only)."""

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    intent: IntentCfg = IntentCfg()


@configclass
class RiderEventCfg(EventCfg):
    """Events of TwbEnvCfg + the rider body."""

    # the rider sits upright at the start of an episode
    reset_rider_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": _hip(), "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    # every policy step (interval 0 s): the rider leans towards the speed they want
    rider_body = EventTerm(
        func=mdp.rider_body_step,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": _hip(),
            "command_name": "base_velocity",
            "lean_per_speed": 0.26,     # rad per m/s  -> 15 deg at 1 m/s
            "lean_max": 0.35,           # rad
            "lean_tau": (0.25, 0.6),    # s
            "sway": 0.0,                # rad of random body movement that is not a command; try 0.03 later
        },
    )


@configclass
class TwbRiderEnvCfg(TwbEnvCfg):
    """Chair with a leaning rider body."""

    observations: RiderObservationsCfg = RiderObservationsCfg()
    events: RiderEventCfg = RiderEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = BALANCE_BOT_RIDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # the rider mass now goes on the rider body (1 kg in the URDF + 40-120 kg here), from the first iteration
        # rider mass split like a seated person (Dempster / Winter): head, trunk and arms lean (53.6 %),
        # pelvis and legs sit on the chair (46.4 %). Genny Zero: users of 40-120 kg.
        self.events.rider_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["rider", "rider_legs"])
        self.events.rider_mass.params["mass_fractions"] = {"rider": 0.536, "rider_legs": 0.464}
        self.events.rider_mass.params["mass_start"] = (40.0, 120.0)
        self.curriculum.rider_gate.params["start_kg"] = 40.0
        self.curriculum.rider_gate.params["max_kg"] = 120.0
        # first version: the only CoM shift is the rider leaning. Fixed random offsets can be added again later with
        # person_com_offset on body "rider".
        self.events.person_com_offset = None
        # posture reward: weight above the axle instead of "chair upright"
        self.rewards.flat_orientation_l2 = None
        self.rewards.balance_pose_l2 = RewTerm(func=mdp.balance_pose_l2, weight=-0.5)
