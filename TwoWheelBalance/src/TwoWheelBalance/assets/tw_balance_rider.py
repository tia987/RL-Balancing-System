"""Articulation config for the two-wheeled chair with a seated mannequin (rider task).

Same chair and wheels as BALANCE_BOT_CFG (tw_balance.py: 55 kg body, 2 x 5 kg wheels, 144 N.m, 18.5 rad/s) plus:
  - URDF data/tw_balance_rider.urdf (built by build_rider_model.py): chair shaped after the Genny Zero
    (seat 64 cm above the floor, backrest, steering column with two grips, footplate) and a 1.75 m mannequin with
    human proportions. Visuals are low-poly meshes in data/meshes/. Set TWB_RIDER_SIMPLE=1 to load
    data/tw_balance_rider_simple.urdf instead (same physics, visuals made of boxes / cylinders / spheres only).
  - two rider links: "rider_legs" (pelvis + legs, fixed to the chair) and "rider" (head, trunk, arms) on the hip
    joint "rider_joint". merge_fixed_joints=False keeps rider_legs as its own body, so the rider_mass event can put
    46.4 % of the rider mass there and 53.6 % on the upper body.
  - actuator "rider": the hip is a spring-damper towards the lean the simulated person aims for
    (target set every step by mdp.rider_body_step).
"""

import os

from isaaclab.actuators import ImplicitActuatorCfg

from .tw_balance import BALANCE_BOT_CFG

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_URDF = "tw_balance_rider_simple.urdf" if os.environ.get("TWB_RIDER_SIMPLE", "0") == "1" else "tw_balance_rider.urdf"

BALANCE_BOT_RIDER_CFG = BALANCE_BOT_CFG.replace(
    spawn=BALANCE_BOT_CFG.spawn.replace(asset_path=os.path.join(_DATA_DIR, _URDF), merge_fixed_joints=False),
    actuators={
        # unchanged: wheels are driven by raw torque commands
        "wheels": BALANCE_BOT_CFG.actuators["wheels"],
        # the person holds the upper body like a spring-damper (N*m/rad, N*m*s/rad). Gravity on a leaning upper body
        # of a 120 kg rider is 65 kg * 9.81 * 0.345 m = 220 N*m/rad, so it sags about 10 % past the aimed angle.
        "rider": ImplicitActuatorCfg(
            joint_names_expr=["rider_joint"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=10.0,
            stiffness=2000.0,
            damping=200.0,
        ),
    },
)
"""Two-wheeled balancing chair with a seated, leaning mannequin."""
