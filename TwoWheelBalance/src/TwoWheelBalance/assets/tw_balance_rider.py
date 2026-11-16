"""Articulation config for the two-wheeled chair with a rider body (shared-control baseline).

Same robot as BALANCE_BOT_CFG (tw_balance.py) plus:
  - URDF data/tw_balance_rider.urdf: adds the "rider" link on a hip joint ("rider_joint") and nicer visuals
  - actuator "rider": the hip joint is a spring-damper towards a target angle. The target is set every step by
    mdp.rider_body_step (how far the simulated person wants to lean).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from .tw_balance import BALANCE_BOT_CFG

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

BALANCE_BOT_RIDER_CFG = BALANCE_BOT_CFG.replace(
    spawn=BALANCE_BOT_CFG.spawn.replace(asset_path=os.path.join(_DATA_DIR, "tw_balance_rider.urdf")),
    actuators={
        # unchanged: wheels are driven by raw torque commands
        "wheels": BALANCE_BOT_CFG.actuators["wheels"],
        # the person holds their upper body like a spring-damper (stiffness N*m/rad, damping N*m*s/rad).
        # Stiff enough that a 120 kg rider does not sag more than about 10 % past the angle they aim for
        # (gravity on the leaning torso is 120 kg * 9.81 * 0.28 m = 330 N*m/rad), damped so it does not swing.
        "rider": ImplicitActuatorCfg(
            joint_names_expr=["rider_joint"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=10.0,
            stiffness=3000.0,
            damping=250.0,
        ),
    },
)
"""Two-wheeled balancing chair with a leaning rider (torque-controlled wheels, spring-damper hip)."""

