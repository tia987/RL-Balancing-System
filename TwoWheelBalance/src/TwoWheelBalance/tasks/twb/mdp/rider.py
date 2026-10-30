import torch

from isaaclab.managers import SceneEntityCfg

def _to_torch(x) -> torch.Tensor:
    """Isaac Lab 3.0 may return Warp arrays instead of torch tensors."""
    if isinstance(x, torch.Tensor):
        return x
    import warp as wp
    return wp.to_torch(x)

def _view(asset):
    view = getattr(asset, "root_physx_view", None)
    return view if view is not None else asset.root_view

def _get_masses(asset) -> torch.Tensor:
    return _to_torch(_view(asset).get_masses()).clone()

def _get_inertias(asset) -> torch.Tensor:
    return _to_torch(_view(asset).get_inertias()).clone()

def _set(asset, masses, inertias, env_ids_cpu):
    view = _view(asset)
    try:
        view.set_masses(masses, env_ids_cpu)
        view.set_inertias(inertias, env_ids_cpu)
    except Exception:                                           # this backend wants Warp arrays
        import warp as wp
        ids = wp.from_torch(env_ids_cpu.to(torch.int32))
        view.set_masses(wp.from_torch(masses.contiguous()), ids)
        view.set_inertias(wp.from_torch(inertias.contiguous()), ids)

def rider_mass_curriculum(env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, mass_start: tuple[float, float] = (5.0, 30.0), mass_end: tuple[float, float] = (40.0, 120.0), ramp_steps: int=48_000):
    """Add a random rider mass [kg] to the bodies in asset_cfg (e.g. base_link) for the resetting envs.

    Range of the uniform sampling:
    - time ramp (default): lower/upper bound move linearly from mass_start to mass_end over
      `ramp_steps` policy steps (24 steps per PPO iteration -> 48_000 steps = 2000 iterations);
    - gated: if rider_mass_gate is active, the upper bound is env.rider_mass_max instead.
    Inertia is scaled with the body mass, so a heavier body is also harder to rotate.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # On the first call, initialise to the environment the rider's mass
    if not hasattr(env, "rider_mass"):
        env.rider_mass = torch.zeros(env.num_envs, device=env.device)
        env._twb_base_masses = _get_masses(asset)
        env._twb_base_inertias = _get_inertias(asset)

    # On the first call, initialise the rider's mass max, that can become over the training
    if hasattr(env, "rider_mass_max"):
        lo, hi = mass_start[0], float(env.rider_mass_max)
    else:
        a = min(env.common_step_counter / max(ramp_steps, 1), 1.0)
        lo = mass_start[0] + a * (mass_end[0] - mass_start[0])
        hi = mass_start[1] + a * (mass_end[1] - mass_start[1])
    hi = max(hi, lo)

    m = torch.empty(len(env_ids), device=env.device).uniform_(lo, hi)
    env.rider_mass[env_ids] = m

    ids_cpu = env_ids.cpu()
    dev = env._twb_base_masses.device                         # CPU or GPU, depending on the backend
    ids = env_ids.to(dev)
    body_ids = asset_cfg.body_ids if isinstance(asset_cfg.body_ids, list) else list(range(env._twb_base_masses.shape[1]))
    bids = torch.tensor(body_ids, device=dev)
    masses = env._twb_base_masses.clone()
    inertias = env._twb_base_inertias.clone()
    base = env._twb_base_masses[ids][:, bids]
    new = base + m.to(dev).unsqueeze(1)
    masses[ids[:, None], bids] = new
    # scale inertia with mass
    ratio = (new / base).unsqueeze(-1)
    inertias[ids[:, None], bids] = env._twb_base_inertias[ids][:, bids] * ratio
    _set(asset, masses, inertias, ids_cpu)

def _episode_success(env, env_ids, command_name, track_term):
    """Per-env 0..1 score of the episodes that are ending now (called before the managers reset).

    Preferred: exactly the definition behind TensorBoard's Metrics/success_rate (UniformVelocityCommand:
    episode-mean xy error < vel_xy_success_threshold AND yaw error < vel_yaw_success_threshold).
    Fallback (older Isaac Lab without those buffers): normalised velocity-tracking reward of the episode.
    """
    term = env.command_manager.get_term(command_name)
    if all(hasattr(term, a) for a in ("_error_xy_sum", "_error_yaw_sum", "_step_count")):
        n = term._step_count[env_ids].clamp_min(1.0)
        ok_xy = term._error_xy_sum[env_ids] / n < term.cfg.vel_xy_success_threshold
        ok_yaw = term._error_yaw_sum[env_ids] / n < term.cfg.vel_yaw_success_threshold
        return (ok_xy & ok_yaw).float()
    rm = env.reward_manager
    ep_sum = rm._episode_sums[track_term][env_ids]
    return (ep_sum / (rm.get_term_cfg(track_term).weight * env.max_episode_length_s)).clamp(0.0, 1.0)



def rider_mass_gate( env, env_ids: torch.Tensor, start_kg: float = 10.0, max_kg: float = 90.0, step_kg: float = 1.0, every_steps: int = 240, min_episodes: int = 150, heavy_frac: float = 0.7, up_success: float = 0.85, down_success: float = 0.6, max_fall_rate: float = 0.05, command_name: str = "base_velocity", track_term: str = "track_lin_vel_xy_exp", fall_term: str = "bad_orientation"):
    if not hasattr(env, "rider_mass_max"):
        env.rider_mass_max = start_kg
    env._gate_n, env._gate_succ, env._gate_falls = 0, 0.0, 0.0
    env._gate_last = env.common_step_counter
    if isinstance(env_ids, slice) or env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) > 0 and hasattr(env, "rider_mass"):
        heavy = env.rider_mass[env_ids] >= heavy_frac * env.rider_mass_max   # mass of the episode that ends now
        ids = env_ids[heavy]
        if len(ids) > 0:
            env._gate_n += len(ids)
            env._gate_succ += _episode_success(env, ids, command_name, track_term).sum().item()
            env._gate_falls += env.termination_manager.get_term(fall_term)[ids].float().sum().item()
    if env.common_step_counter - env._gate_last >= every_steps and env._gate_n >= min_episodes:
        succ = env._gate_succ / env._gate_n
        fall_rate = env._gate_falls / env._gate_n
        if succ > up_success and fall_rate < max_fall_rate:
            env.rider_mass_max = min(env.rider_mass_max + step_kg, max_kg)
        elif succ < down_success:
            env.rider_mass_max = max(env.rider_mass_max - step_kg, start_kg)
        env.rider_mass_gate_score = succ                        # for logging, see rider_gate_score
        env._gate_n, env._gate_succ, env._gate_falls = 0, 0.0, 0.0
        env._gate_last = env.common_step_counter
    return torch.tensor(float(env.rider_mass_max))

def rider_gate_score(env, env_ids: torch.Tensor) -> torch.Tensor:
    """Logging only: success rate of the heavy-rider episodes at the last gate decision (Curriculum/<term>)."""
    return torch.tensor(float(getattr(env, "rider_mass_gate_score", 0.0)))

def rider_mass_obs(env, scale: float=0.01) -> torch.Tensor:
    """Rider mass per env (scaled: 100 kg -> 1.0). Privileged: only the teacher / critic may see it."""
    if not hasattr(env, "rider_mass"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return (env.rider_mass * scale).unsqueeze(1)

def rider_mass_mean(env, env_ids: torch.Tensor) -> torch.Tensor:
    """Curriculum term used only for logging: mean rider mass in kg (TensorBoard: Curriculum/rider_mass)."""
    if not hasattr(env, "rider_mass"):
        return torch.tensor(0.0)
    return env.rider_mass.mean()
