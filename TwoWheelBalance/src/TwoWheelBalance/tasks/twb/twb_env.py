"""TwoWheelBalance environment = Isaac Lab's ManagerBasedRLEnv + two optional tools for `isaaclab play`:

1) TWB_OVERLAY=1  text box on recorded videos (rider mass, commanded vs actual velocity, pitch, torques, terrain level)
                  for the robot the camera follows. TWB_OVERLAY_ENV=<index> chooses another robot.

2) TWB_EVAL=1     measurement of ALL robots instead of judging one robot in a video. After TWB_EVAL_STEPS policy steps
                  (default 1100, just below the 1200 of the videos) it prints a table and writes a CSV into the run
                  folder (eval_*.csv): per robot the rider mass, CoM offset, commanded and actual speed, pitch, torque,
                  torque sign flips per second and falls, plus a summary (speed per rider-mass group, and how much of
                  the speed error is explained by the CoM offset).
                  Samples are only counted when the command has been constant for TWB_EVAL_WARMUP seconds (default 4).

Put this file in  TwoWheelBalance/src/TwoWheelBalance/tasks/twb/twb_env.py  (entry point in config/twb/__init__.py:
entry_point="TwoWheelBalance.tasks.twb.twb_env:TwbEnv"). Training is unchanged: without the environment variables
this class does nothing extra.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv


def _t(x):
    """Isaac Lab 3.0 data fields are proxy arrays with a .torch view."""
    return x.torch if hasattr(x, "torch") else x


class TwbEnv(ManagerBasedRLEnv):
    def step(self, action):
        if not getattr(self, "_twb_ready", False):
            self._twb_ready = True
            if os.environ.get("TWB_OVERLAY", "0") == "1":
                for rec in getattr(self, "video_recorders", []):
                    grab = rec._get_frame
                    rec._get_frame = lambda grab=grab: self._twb_overlay(grab())
            self._twb_eval_on = os.environ.get("TWB_EVAL", "0") == "1"
        out = super().step(action)
        if self._twb_eval_on:
            try:
                self._twb_eval_step()
            except Exception as e:  # never break a run because of the measurement
                print(f"[TWB eval] disabled: {e!r}")
                self._twb_eval_on = False
        return out

    def close(self):
        if getattr(self, "_twb_eval_on", False) and not getattr(self, "_twb_eval_done", False):
            try:
                self._twb_eval_report()
            except Exception as e:
                print(f"[TWB eval] report failed: {e!r}")
        return super().close()

    def _twb_wheel_joints(self) -> list[int]:
        """Indices of the two wheel joints (the rider task has a third joint, the rider's hip)."""
        if not hasattr(self, "_twb_wj"):
            try:
                self._twb_wj = [int(j) for j in self.scene["robot"].find_joints(".*_wheel_joint")[0]][:2]
            except Exception:
                self._twb_wj = [0, 1]
        return self._twb_wj

    # ------------------------------------------------------------------ measurement of all robots
    def _twb_eval_step(self):
        robot = self.scene["robot"]
        n, dev = self.num_envs, self.device
        if not hasattr(self, "_ev"):
            self._ev = {k: torch.zeros(n, device=dev) for k in
                        ("n", "cmd", "vx", "abs_err", "wz_err", "pitch", "pitch2", "tau", "flips", "falls", "rider")}
            self._ev_prev_tau = torch.zeros(n, device=dev)
            self._ev_prev_cmd = torch.zeros(n, 3, device=dev)
            self._ev_const = torch.zeros(n, device=dev)                  # seconds the command has been constant
            self._ev_steps = 0
            self._ev_warm = float(os.environ.get("TWB_EVAL_WARMUP", "4"))
            self._ev_total = int(os.environ.get("TWB_EVAL_STEPS", "1100"))
        ev = self._ev
        cmd = self.command_manager.get_command("base_velocity")
        changed = (cmd - self._ev_prev_cmd).abs().sum(1) > 1e-4
        self._ev_prev_cmd = cmd.clone()
        new_episode = self.episode_length_buf == 0                      # robot was reset in this step
        self._ev_const = torch.where(changed | new_episode, torch.zeros_like(self._ev_const), self._ev_const + self.step_dt)
        ev["falls"] += self.reset_terminated.float()
        ok = (self._ev_const >= self._ev_warm).float()
        v = _t(robot.data.root_lin_vel_b)
        w = _t(robot.data.root_ang_vel_b)
        g = _t(robot.data.projected_gravity_b)
        tau = _t(robot.data.applied_torque)[:, self._twb_wheel_joints()].mean(1)
        pitch = torch.atan2(g[:, 0], -g[:, 2])
        ev["n"] += ok
        ev["cmd"] += ok * cmd[:, 0]
        ev["vx"] += ok * v[:, 0]
        ev["abs_err"] += ok * (cmd[:, 0] - v[:, 0]).abs()
        ev["wz_err"] += ok * (cmd[:, 2] - w[:, 2]).abs()
        ev["pitch"] += ok * pitch
        ev["pitch2"] += ok * pitch**2
        ev["tau"] += ok * tau.abs()
        ev["flips"] += ok * ((tau * self._ev_prev_tau) < 0).float()
        if hasattr(self, "rider_mass"):
            ev["rider"] += ok * self.rider_mass
        self._ev_prev_tau = tau.clone()
        self._ev_steps += 1
        if self._ev_steps == self._ev_total:
            self._twb_eval_report()

    def _twb_eval_report(self):
        self._twb_eval_done = True
        if not hasattr(self, "_ev"):
            return
        ev = {k: x.detach().cpu().numpy().astype(np.float64) for k, x in self._ev.items()}
        n = np.maximum(ev["n"], 1.0)
        seen = ev["n"] > 30                                             # robots with at least 0.5 s of valid samples
        cmd, vx = ev["cmd"] / n, ev["vx"] / n
        pitch = np.degrees(ev["pitch"] / n)
        pitch_sd = np.degrees(np.sqrt(np.maximum(ev["pitch2"] / n - (ev["pitch"] / n) ** 2, 0.0)))
        tau, flips = ev["tau"] / n, ev["flips"] / (n * self.step_dt)
        rider = ev["rider"] / n
        abs_err, wz_err = ev["abs_err"] / n, ev["wz_err"] / n
        com = np.full((self.num_envs, 3), np.nan)
        try:
            robot = self.scene["robot"]
            b = robot.find_bodies("base_link")[0][0]
            com = _t(robot.data.body_com_pose_b)[:, b, :3].detach().cpu().numpy().astype(np.float64)
        except Exception as e:
            print(f"[TWB eval] CoM offset not available: {e!r}")
        # --- CSV with one row per robot
        folder = getattr(self.cfg, "log_dir", None) or "."
        path = os.path.join(folder, time.strftime("eval_%Y-%m-%d_%H-%M-%S.csv"))
        try:
            with open(path, "w") as f:
                f.write("robot,rider_kg,com_x_cm,com_z_cm,cmd_vx,vx,abs_err,wz_abs_err,pitch_deg,pitch_sd_deg,abs_torque,flips_per_s,falls,samples\n")
                for i in range(self.num_envs):
                    f.write(f"{i},{rider[i]:.1f},{com[i,0]*100:.2f},{com[i,2]*100:.2f},{cmd[i]:.3f},{vx[i]:.3f},{abs_err[i]:.3f},"
                            f"{wz_err[i]:.3f},{pitch[i]:.2f},{pitch_sd[i]:.2f},{tau[i]:.2f},{flips[i]:.2f},{int(ev['falls'][i])},{int(ev['n'][i])}\n")
        except OSError as e:
            path = f"(not written: {e})"
        # --- summary
        L = ["", "=" * 100, f"[TWB eval] {int(seen.sum())}/{self.num_envs} robots measured, {self._ev_steps} steps "
             f"({self._ev_steps * self.step_dt:.0f} s), per-robot table: {path}"]
        if seen.sum() >= 2:
            def line(name, m):
                return (f"{name:<22s} robots {int(m.sum()):4d} | cmd {cmd[m].mean():+5.2f} | speed {vx[m].mean():+5.2f} +- {vx[m].std():4.2f} "
                        f"(min {vx[m].min():+5.2f}, max {vx[m].max():+5.2f}) | |error| {abs_err[m].mean():4.2f} | pitch sd {pitch_sd[m].mean():4.1f} deg | "
                        f"|torque| {tau[m].mean():5.1f} N.m | flips {flips[m].mean():4.1f}/s | falls {int(ev['falls'][m].sum())}")
            L.append(line("all robots", seen))
            if rider[seen].max() - rider[seen].min() > 10.0:
                edges = np.arange(0.0, rider[seen].max() + 20.0, 20.0)
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = seen & (rider >= lo) & (rider < hi)
                    if m.sum() > 0:
                        L.append(line(f"rider {lo:3.0f}-{hi:3.0f} kg", m))
            err = (cmd - vx)[seen]
            for name, x, unit in (("CoM offset x", com[seen, 0] * 100, "cm"), ("mean pitch (lean)", pitch[seen], "deg")):
                if np.all(np.isfinite(x)) and x.std() > 1e-6 and err.std() > 1e-6:
                    slope, r = np.polyfit(x, err, 1)[0], np.corrcoef(x, err)[0, 1]
                    L.append(f"speed error vs {name:<18s}: {slope:+.3f} m/s per {unit}, correlation r = {r:+.2f} "
                             f"(r^2 = {r*r:.2f} of the robot-to-robot differences)")
        L.append("=" * 100)
        print("\n".join(L), flush=True)

    # ------------------------------------------------------------------ overlay
    def _twb_env_index(self) -> int:
        if "TWB_OVERLAY_ENV" in os.environ:
            return int(os.environ["TWB_OVERLAY_ENV"]) % self.num_envs
        viz = getattr(getattr(self.cfg, "sim", None), "default_visualizer_cfg", None)
        idx = getattr(viz, "origin_env_index", None)
        if idx is None:
            idx = getattr(getattr(self.cfg, "viewer", None), "env_index", 0)
        return int(idx) % self.num_envs

    def _twb_lines(self) -> list[str]:
        i = self._twb_env_index()
        robot = self.scene["robot"]
        lines = [f"robot #{i}"]
        if hasattr(self, "rider_mass"):
            lines.append(f"rider      {self.rider_mass[i].item():6.1f} kg")
        if hasattr(self, "rider_lean_target"):                              # task with a rider body
            try:
                hip = robot.find_joints("rider_joint")[0][0]
                lean = math.degrees(_t(robot.data.joint_pos)[i, hip].item())
                lines.append(f"rider lean {lean:+5.1f} deg (aims {math.degrees(self.rider_lean_target[i].item()):+5.1f})")
            except Exception:
                pass
        cmd = self.command_manager.get_command("base_velocity")[i]
        v = _t(robot.data.root_lin_vel_b)[i]
        w = _t(robot.data.root_ang_vel_b)[i]
        lines.append(f"cmd    vx {cmd[0].item():+5.2f} m/s   wz {cmd[2].item():+5.2f} rad/s")
        lines.append(f"actual vx {v[0].item():+5.2f} m/s   wz {w[2].item():+5.2f} rad/s")
        g = _t(robot.data.projected_gravity_b)[i]
        pitch = math.degrees(math.atan2(g[0].item(), -g[2].item()))
        lines.append(f"pitch      {pitch:+5.1f} deg")
        wj = self._twb_wheel_joints()
        tau = _t(robot.data.applied_torque)[i]
        lines.append(f"torque  L {tau[wj[0]].item():+5.0f}  R {tau[wj[1]].item():+5.0f} N·m")
        terrain = getattr(self.scene, "terrain", None)
        levels = getattr(terrain, "terrain_levels", None)
        if levels is not None:
            lines.append(f"terrain level {int(levels[i].item())}")
        return lines

    def _twb_overlay(self, frame):
        if frame is None:
            return frame
        try:
            return draw_text_box(frame, self._twb_lines())
        except Exception as e:  # never lose a video because of the overlay
            if not getattr(self, "_twb_overlay_warned", False):
                print(f"[TWB overlay] disabled: {e!r}")
                self._twb_overlay_warned = True
            return frame


def draw_text_box(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw white text on a semi-transparent dark box in the top-left corner of an RGB(A) uint8 frame."""
    from PIL import Image, ImageDraw, ImageFont

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    img = Image.fromarray(arr[..., :3]).convert("RGBA")
    size = max(14, img.height // 32)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    box = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(box)
    line_h = int(size * 1.3)
    width = max(draw.textlength(s, font=font) for s in lines)
    draw.rectangle([10, 10, 30 + width, 20 + line_h * len(lines)], fill=(0, 0, 0, 150))
    for k, s in enumerate(lines):
        draw.text((20, 15 + k * line_h), s, font=font, fill=(255, 255, 255, 255))
    out = np.array(Image.alpha_composite(img, box).convert("RGB"))
    return out if arr.shape[-1] == 3 else np.concatenate([out, arr[..., 3:]], axis=-1)

