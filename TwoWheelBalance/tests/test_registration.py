# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

import TwoWheelBalance.tasks  # noqa: F401


def test_task_registrations():
    """The project tasks must expose valid environment and agent entry points."""
    expected = {
        "Twowheelbalance-Twb-Twb": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "TwoWheelBalance.tasks.twb.config.twb.env_cfg:TwbEnvCfg",
            "default_agent": "rsl_rl",
        },
        "Twowheelbalance-Twb-Twb-Direct": {
            "entry_point": "TwoWheelBalance.tasks.twb_direct.config.twb.env:TwbEnv",
            "env_cfg_entry_point": "TwoWheelBalance.tasks.twb_direct.config.twb.env_cfg:TwbEnvCfg",
            "default_agent": "rsl_rl",
        },
        "Twowheelbalance-Twb-Marl-Twb-Direct": {
            "entry_point": "TwoWheelBalance.tasks.twb_marl_direct.config.twb.env:TwbMarlEnv",
            "env_cfg_entry_point": "TwoWheelBalance.tasks.twb_marl_direct.config.twb.env_cfg:TwbMarlEnvCfg",
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]
