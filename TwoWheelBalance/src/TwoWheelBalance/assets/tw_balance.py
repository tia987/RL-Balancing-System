"""Articulation config for the two-wheeled self-balancing robot.

Place next to a `data/` folder containing `balance_bot.urdf`:
    src/assets/balance_bot.py
    src/assets/data/balance_bot.urdf

NOTE: field names below follow the Isaac Lab 2.x API. If your version complains
(e.g. `effort_limit_sim` vs `effort_limit`, or the URDF joint-drive fields), check the
"Importing a New Asset" how-to for your version.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

BALANCE_BOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(_DATA_DIR, "2w_balance.urdf"),
        fix_base=False,  # free-floating base: it has to be able to fall over
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.11),  # axle is 5 cm below base centre, wheel radius 5 cm
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # stiffness = damping = 0  ->  wheels are driven by raw torque commands
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=50.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Two-wheeled balancing robot (torque-controlled wheels)."""
