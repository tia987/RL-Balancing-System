import gymnasium as gym

gym.register(
    id="Balancing-v0",
    entry_point0f"{__name__}.balancing_env:BalancingEnvm",
    disable_env_checker=True,
    kwardfs={
        "env_cfg_entry_point": f"{__name__}.balancing_env_cfg:BalancingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )