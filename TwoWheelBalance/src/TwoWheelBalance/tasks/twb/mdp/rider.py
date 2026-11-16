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
    # keep our own copy of the current body masses (used by balance_error): rows of the robots that were just reset
    if not hasattr(env, "_twb_masses"):
        env._twb_masses = env._twb_base_masses.clone().to(env.device)
    env._twb_masses[env_ids[:, None], bids.to(env.device)] = new.to(env.device)

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


def rider_mass_gate(
    env,
    env_ids: torch.Tensor,
    start_kg: float = 10.0,
    max_kg: float = 120.0,
    step_kg: float = 2.0,
    every_steps: int = 240,
    min_episodes: int = 150,
    heavy_frac: float = 0.7,
    up_success: float = 0.85,
    down_success: float = 0.6,
    max_fall_rate: float = 0.05,
    command_name: str = "base_velocity",
    track_term: str = "track_lin_vel_xy_exp",
    fall_term: str = "bad_orientation",
):
    """Performance-gated maximum rider mass [kg] (use together with rider_mass_curriculum).

    - Isaac Lab calls curriculum terms at EVERY env step that has resets, so the gate only collects the
      episodes that end and decides every `every_steps` env steps (240 = 10 PPO iterations at 24 steps/iter).
    - Only the "heavy" episodes count: rider mass >= heavy_frac * current max. Light riders are easy and
      would hide the failures; it also removes most of the lag from 20 s long episodes.
    - success = same as Metrics/success_rate (velocity tracked well over the episode); falls = bad_orientation.
    - up   : success > up_success and falls < max_fall_rate  -> max + step_kg
      down : success < down_success                          -> max - step_kg (not below start_kg)
    """
    if not hasattr(env, "rider_mass_max"):
        env.rider_mass_max = start_kg
        env._gate_stats = torch.zeros(3, device=env.device)      # [n heavy episodes, successes, falls]
        env._gate_last = env.common_step_counter
    if isinstance(env_ids, slice) or env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) > 0 and hasattr(env, "rider_mass"):
        # masks instead of boolean indexing: no GPU->CPU sync at every step (keeps training fast)
        heavy = (env.rider_mass[env_ids] >= heavy_frac * env.rider_mass_max).float()
        succ = _episode_success(env, env_ids, command_name, track_term)
        falls = env.termination_manager.get_term(fall_term)[env_ids].float()
        env._gate_stats += torch.stack([heavy.sum(), (heavy * succ).sum(), (heavy * falls).sum()])
    if env.common_step_counter - env._gate_last >= every_steps:
        n, n_succ, n_falls = env._gate_stats.tolist()                # one sync per decision window
        if n >= min_episodes:
            succ_rate, fall_rate = n_succ / n, n_falls / n
            if succ_rate > up_success and fall_rate < max_fall_rate:
                env.rider_mass_max = min(env.rider_mass_max + step_kg, max_kg)
            elif succ_rate < down_success:
                env.rider_mass_max = max(env.rider_mass_max - step_kg, start_kg)
            env.rider_mass_gate_score = succ_rate                     # for logging, see rider_gate_score
            env._gate_stats.zero_()
            env._gate_last = env.common_step_counter
        else:                                                         # not enough heavy episodes yet:
            env._gate_last += every_steps // 4                        # keep counting, check again a bit later
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

def rider_com_obs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link"), scale: float = 10.0) -> torch.Tensor:
    """Centre-of-mass offset of the body with the rider: x (fore-aft), y (sideways), z (up), in m * scale (0.1 m -> 1.0).

    Privileged, like rider_mass_obs: a real robot cannot measure it directly. It is the value set by the
    person_com_offset event (randomize_rigid_body_com). A robot whose CoM is 3 cm in front of / behind the axle has to
    stand tilted by 3-5 degrees to balance; without this number (or a memory) the policy cannot tell that tilt apart
    from "I am accelerating", and drives too slow or too fast.
    """
    asset = env.scene[asset_cfg.name]
    com = asset.data.body_com_pose_b
    com = com.torch if hasattr(com, "torch") else com
    body = asset_cfg.body_ids[0] if isinstance(asset_cfg.body_ids, list) else 0
    return com[:, body, :3] * scale


# ----------------------------------------------------------------------------------------------------------------------
# Balance pose: where the chair has to stand so that it does not fall, for THIS rider
# ----------------------------------------------------------------------------------------------------------------------
def _t(x):
    """Isaac Lab 3.0 data fields are proxy arrays with a .torch view."""
    return x.torch if hasattr(x, "torch") else x


def _body_masses(env, asset) -> torch.Tensor:
    """Current mass of every body, shape (num_envs, num_bodies), on env.device."""
    if hasattr(env, "_twb_masses"):
        return env._twb_masses
    return _t(asset.data.body_mass).to(env.device)


def balance_error(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), wheel_bodies: str = ".*_wheel") -> torch.Tensor:
    """Angle [rad] between "straight up" and the line from the wheel axle to the centre of mass of everything that
    sits on the axle (chair body + rider + any CoM shift), measured in the driving direction.

    0   -> the weight is exactly above the axle: the chair is balanced, whatever its own tilt is
    > 0 -> the weight is in front of the axle: the chair falls forward unless it accelerates
    It is computed from the simulator's centre-of-mass positions, so it automatically includes the rider mass, the
    random CoM offset and a moving rider body. Shape (num_envs,).
    """
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "_twb_wheel_ids"):
        wheels = list(asset.find_bodies(wheel_bodies)[0])
        env._twb_wheel_ids = torch.tensor(wheels, device=env.device)
        env._twb_upper_ids = torch.tensor([i for i in range(asset.num_bodies) if i not in wheels], device=env.device)
    com = _t(asset.data.body_com_pos_w)[..., :3]                              # (N, bodies, 3) world
    mass = _body_masses(env, asset)[:, env._twb_upper_ids].unsqueeze(-1)      # (N, upper, 1)
    axle = com[:, env._twb_wheel_ids].mean(dim=1)
    upper = (com[:, env._twb_upper_ids] * mass).sum(dim=1) / mass.sum(dim=1)
    r = upper - axle
    yaw = _t(asset.data.heading_w)
    forward = r[:, 0] * torch.cos(yaw) + r[:, 1] * torch.sin(yaw)
    return torch.atan2(forward, r[:, 2])


def balance_pose_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), wheel_bodies: str = ".*_wheel") -> torch.Tensor:
    """Posture penalty that replaces flat_orientation_l2: (balance_error)^2 + (sideways tilt)^2.

    flat_orientation_l2 asks the chair body to be upright. With a rider whose weight is not above the axle that pose
    is not a balance point. This term asks for "weight above the axle" instead = zero pitch + the offset of this rider.
    Without any CoM shift both terms are the same.
    """
    asset = env.scene[asset_cfg.name]
    roll = _t(asset.data.projected_gravity_b)[:, 1]
    return torch.square(balance_error(env, asset_cfg, wheel_bodies)) + torch.square(roll)


def balance_pitch_obs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), wheel_bodies: str = ".*_wheel") -> torch.Tensor:
    """Privileged observation: the pitch [rad] the chair body must have right now to be balanced (the "offset").
    0 for a centred rider; about -0.07 rad (-4 deg) if the weight sits 3 cm in front of the axle. Shape (num_envs, 1)."""
    asset = env.scene[asset_cfg.name]
    g = _t(asset.data.projected_gravity_b)
    pitch = torch.atan2(g[:, 0], -g[:, 2])
    return (pitch - balance_error(env, asset_cfg, wheel_bodies)).unsqueeze(1)


