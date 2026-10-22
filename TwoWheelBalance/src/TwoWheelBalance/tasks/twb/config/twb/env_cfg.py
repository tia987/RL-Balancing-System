# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from TwoWheelBalance.tasks.twb.rough_terrain import TWB_TERRAINS_CFG
# from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from TwoWheelBalance.tasks.twb import mdp

##
# Pre-defined configs
##

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from TwoWheelBalance.assets.tw_balance import BALANCE_BOT_CFG


@configclass
class TwbPhysicsCfg(PresetCfg):
    """Physics presets for the cart-pole environment."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=100,
            nconmax=10,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


##
# Scene definition
##


@configclass
class TwbSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # Define the robot
    robot: ArticulationCfg = BALANCE_BOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Imported from anymal velocity_env_cfg.py 
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator", #previous: generator
        terrain_generator=TWB_TERRAINS_CFG.replace(curriculum=True),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    # robot: ArticulationCfg = MISSING
    # sensors: the concrete implementation is selected automatically from the active physics backend;
    # backend-specific fields such as ``global_world_only`` are ignored by the backends that do not use them
    
    # @TODO: For the master thesis
    #
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    #     global_world_only=True,
    # )
    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True) # Remove contact force for the moment, given that robot doesn't have sensors
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.8,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP. Set wheel joints to rotate"""
    
    # @brief: Torque
    # @CHANGE: Reduced scale from 100 to 20
    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*_wheel_joint"], scale=20.0)
    # @brief: Angular velocity instead of torque
    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*_wheel_joint"], scale=10.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # Added noise to training for the observation
        # @NOTE: probably base_lin_vel to remove due to hardware
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel) # Case without noise
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # TODO: check difference between with and without _rel, @note: anymal doesn't use _rel
        # setup for the moment without noise
        #
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # @ADDED: previous action, from anymal
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self) -> None:
            self.enable_corruption = True # Toggle noise # Toggle noise
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # @brief: Randomize the body mass between 40 and 100 kg, the robot wheel torque must be adapted to this additional mass
    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            # multiplicative +-25% log-uniform: scale-invariant across robot sizes with geometric mean 1.0
            "mass_distribution_params": (40.0,100.0),
            "operation": "add",
            "distribution": "log_uniform",
        },
    )
    # @brief: Shift the center of mass of the person randomly
    person_com_offset = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "z": (0.0, 0.3)},  # m; rider sits higher
        }
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"pitch": (-0.2, 0.2)}, 
            "velocity_range": {
                "x": (-0.1, 0.1), 
                "y": (-0.1, 0.1), 
                "z": (-0.1, 0.1), 
                "roll": (-0.1, 0.1), # Probably only yaw and pitch
                "pitch": (-0.1, 0.1), 
                "yaw": (-0.1, 0.1)
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # Positive rewards - tasks
    alive = RewTerm(func=mdp.is_alive, weight=0.5)

    # Locomotion
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # @TODO:Check that this function ensures robot vertical stabilization -> Yaw 
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    # Negative rewards - penalties
    #
    # Balance
    # Keep base upright
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.5,
    )
    # Lower wheel velocity
    # wheel_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    # )
    # Lower angular velocity
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Efficiency
    # Force less actions
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # Reduce joint accleration @BRIEF: Not sure about adding it in this case
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    
    # Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0) # Turn it off for the moment @CHECK: if necessary to reactivate later
    



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Robot tilts too much
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": 0.6,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum settings."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
    )

##
# Environment configuration
##

@configclass
class TwbEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cart-pole environment."""

    # Scene settings
    scene: TwbSceneCfg = TwbSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # visualizer camera settings
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0)) 
        # Visualizer to track the robot directly
        # self.viewer = ViewerCfg(
        #     origin_type="asset_root",
        #     asset_name="robot",
        #     env_index=-1,
        #     eye=(3.0, 3.0, 1.5),
        #     lookat=(0.0, 0.0, 0.0),
        # )
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = TwbPhysicsCfg()
